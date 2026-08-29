import numpy as np
import pandas as pd
import scipy.linalg as la

import duckdb
import polars as pl
import polars.selectors as cs
from scipy import stats
import os
from datetime import datetime

import copy

from typing import Literal, Self
from .utils import add_lag_features

class VariableCoefficientModel():
    def __init__(self, index_by):
        self.index_by = index_by
        
    def reset_parameters(self):
        self.x_mats = []               # (N, T, k)
        self.y_stack = []                # (N, T)
        self.beta_mat = []            # (N, k)  <- each row is β̂ᵢ
        self.sigma2_arr = []         # (N,)
        self.residuals_dict = {}
        self.blue_beta_list = []
        self.beta_gls = None
        
    def fit(self, 
            panel_df: pd.DataFrame, 
            regressors: dict[str, list[str]], 
            response: dict[str, list[str]], 
            lags: int = 0, 
            model_type: Literal["random", "fixed"] = "random", 
            intercept: bool = True, 
            asymptotic: bool = True, 
            bayes_estimator: bool = False
        ) -> Self:
        """Public entry point — validates/dispatches to _fit and returns self."""
        self.reset_parameters()
        self.individual_ids = panel_df[self.index_by].unique()
        self.regressors_with_lags = copy.deepcopy(regressors)
        self.response = response
        self.intercept = intercept
        if lags > 0:
            self.panel_df_with_lag = add_lag_features(
                panel_df, self.index_by, None, response["y"][0], lags, drop_nans=True
            )
            for i in self.individual_ids:
                self.regressors_with_lags[f"exog_{i}"] = self.regressors_with_lags[f"exog_{i}"] + [f"Weekly_Sales_L{lag}" for lag in range(1, lags + 1)]
        else:
            self.panel_df_with_lag = panel_df
        self.T = self.panel_df_with_lag[self.index_by].value_counts().iloc[0]
        self.N = len(self.individual_ids)
        self.k = len(regressors["exog_1"])
        if intercept:
            self.k += 1
        self._individual_ols(
            regressors=self.regressors_with_lags,
            response=response,
            intercept=intercept
            
        )
        if model_type == "random":
            return self._fit_swamy(asymptotic, bayes_estimator)
        elif model_type == "fixed":
            return self._fit_sur()

    def _individual_ols(self, regressors, response, intercept):
        x_mats, y_arrs, beta_list, sigma2_list = [], [], [], []

        for i in self.individual_ids:
            mask = self.panel_df_with_lag[self.index_by] == i
            xreg_mat = self.panel_df_with_lag.loc[mask, regressors[f"exog_{i}"]].to_numpy()
            y_i = self.panel_df_with_lag.loc[mask, response["y"][0]].to_numpy()

            if intercept:
                xreg_mat = np.hstack((np.ones((self.T, 1)), xreg_mat))

            XtX_i = xreg_mat.T @ xreg_mat
            beta_i = np.linalg.solve(XtX_i, xreg_mat.T @ y_i)
            resid_i = y_i - xreg_mat @ beta_i
            self.residuals_dict[i] = resid_i
            sigma2_i = resid_i @ resid_i / (self.T - self.k)

            x_mats.append(xreg_mat)
            y_arrs.append(y_i)
            beta_list.append(beta_i)
            sigma2_list.append(sigma2_i)

        self.x_mats = np.stack(x_mats)                # (N, T, k)
        self.y_stack = np.stack(y_arrs)                # (N, T)
        self.beta_mat = np.stack(beta_list)             # (N, k)  <- each row is β̂ᵢ
        self.sigma2_arr = np.array(sigma2_list)         # (N,)
        
    def _fit_sur(self) -> Self:
        residuals_mat = pd.DataFrame(self.residuals_dict).to_numpy()
        residuals_sigma_mat = (residuals_mat.T @ residuals_mat) / self.T
        residuals_sigma_inv_mat = la.inv(residuals_sigma_mat)
        
        # Extract the diagonal variances from Sigma
        diag_variances = np.diag(residuals_sigma_mat)
        
        # Calculate standard deviations (square root of variances)
        residuals_std = np.sqrt(diag_variances)
        
        # Compute the cross-correlation matrix using outer product division
        # This divides every element sigma_ij by (std_i * std_j)
        self.residuals_corr_mat = residuals_sigma_mat / np.outer(residuals_std, residuals_std)
        
        XtX_tensor = np.einsum('itk,jtl->ijkl', self.x_mats, self.x_mats)
        X_Om_X_tensor = residuals_sigma_inv_mat[:, :, np.newaxis, np.newaxis] * XtX_tensor
        X_Om_X = X_Om_X_tensor.transpose(0, 2, 1, 3).reshape(self.N * self.k, self.N * self.k)
        XtY_tensor = np.einsum('itk,jt->ijk', self.x_mats, self.y_stack)
        X_Om_Y_mat = np.einsum('ij,ijk->ik', residuals_sigma_inv_mat, XtY_tensor)
        X_Om_Y = X_Om_Y_mat.reshape(self.N * self.k, 1)
        
        self.beta_SUR_flat = la.solve(X_Om_X, X_Om_Y).flatten()   # <- fix here, or use .ravel()
        self.beta_SUR_mat = self.beta_SUR_flat.reshape(self.N, self.k)

        
        # =========================================================================
        # EXTENSION: HYPOTHESIS TESTING & INFERENCE STATS
        # =========================================================================
        
        # 6. Compute Coefficient Covariance Matrix
        # Var(Beta) = inv(X_Om_X)
        self.beta_cov_mat = la.inv(X_Om_X)
        
        # 7. Extract Standard Errors
        # Take the square root of the diagonal variances
        self.beta_se_flat = np.sqrt(np.diag(self.beta_cov_mat))   # (N*k,)
        self.beta_se_mat = self.beta_se_flat.reshape(self.N, self.k)
        
        # 8. Compute t-statistics
        self.t_stats_flat = self.beta_SUR_flat / self.beta_se_flat   # now (N*k,) / (N*k,) = (N*k,) correctly
        self.t_stats_mat = self.t_stats_flat.reshape(self.N, self.k)
        
        # 9. Compute p-values using the Student-t distribution
        # Degrees of freedom per equation = T - k
        df_residual = self.T - self.k
        self.p_values_flat = 2 * (1 - stats.t.cdf(np.abs(self.t_stats_flat), df=df_residual))
        self.p_values_mat = self.p_values_flat.reshape(self.N, self.k)
        
        
        lower_tri = np.tril_indices(self.N, k=-1)
        r_ij = self.residuals_corr_mat[lower_tri]                                    # off-diagonal correlations, no double counting

        # ---------- 1. Breusch-Pagan LM test ----------
        # H0: Sigma is diagonal (no cross-equation correlation)
        lm_statistic = self.T * np.sum(r_ij ** 2)
        lm_df = int(self.N * (self.N - 1) / 2)
        lm_p_value = stats.chi2.sf(lm_statistic, df=lm_df)

        # ---------- 2. Pesaran (2004) CD test ----------
        # Same H0, but robust to large N relative to T
        cd_statistic = np.sqrt(2 * self.T / (self.N * (self.N - 1))) * np.sum(r_ij)
        cd_p_value = 2 * (1 - stats.norm.cdf(np.abs(cd_statistic)))
        return self, {
                "lm_stat": float(lm_statistic),
                "lm_df": lm_df,
                "lm_p_value": float(lm_p_value),
                "cd_stat": float(cd_statistic),
                "cd_p_value": float(cd_p_value),
            }

        
        
        
    def _fit_swamy(self, asymptotic=True, bayes_estimator=False):
        """Does the actual work: per-unit OLS, Swamy-Arora random coefficients, FGLS."""

        beta_bar = self.beta_mat.mean(axis=0)
        swamy_Delta = 1 / (self.N - 1) * (self.beta_mat.T @ self.beta_mat - self.N * np.outer(beta_bar, beta_bar))

        if asymptotic:
            V_i_sum = np.array([
                self.sigma2_arr[i] * np.linalg.inv(self.x_mats[i].T @ self.x_mats[i])
                for i in range(self.N)
            ]).sum(axis=0)
            swamy_Delta -= 1 / self.N * V_i_sum

        Phi_list, PhiInv_list = [], []
        for i in range(self.N):
            Phi_i = self.x_mats[i] @ swamy_Delta @ self.x_mats[i].T + self.sigma2_arr[i] * np.eye(self.T)
            Phi_list.append(Phi_i)
            PhiInv_list.append(np.linalg.inv(Phi_i))

        self.Phi_list = np.stack(Phi_list)
        self.PhiInv_list = np.stack(PhiInv_list)

        XtPhiIX_sum = np.einsum('itj,its,isl->jl', self.x_mats, self.PhiInv_list, self.x_mats)
        XtPhiIy_sum = np.einsum('itj,its,is->j', self.x_mats, self.PhiInv_list, self.y_stack)

        self.beta_gls = np.linalg.solve(XtPhiIX_sum, XtPhiIy_sum)
        
        
        self.blue_beta_list = []
        for i in self.individual_ids-1:
            blue_beta_i = self.beta_gls + swamy_Delta @ self.x_mats[i].T @ np.linalg.inv(self.x_mats[i] @ swamy_Delta @ self.x_mats[i].T + self.sigma2_arr[i] * np.eye(self.T)) @ (self.y_stack[i] - self.x_mats[i] @ self.beta_gls)
            self.blue_beta_list.append(blue_beta_i)
    
        
    def gen_breusch_pagan(self):
        if np.any(np.isclose(self.sigma2_arr, 0)):
            raise ValueError("Zero noise variance detected.")
        sigma_diag_mat = np.diag(1/self.sigma2_arr**0.5)
        mean_regressors = []
        means_per_unit = self.panel_df_with_lag.groupby(self.index_by)[list(set(val for sublist in self.regressors_with_lags.values() for val in sublist))].mean()
        ybar_vec = self.panel_df_with_lag.groupby(self.index_by)[self.response["y"][0]].mean()
        
        for i in self.individual_ids:
            mean_regressors.append(means_per_unit[self.regressors_with_lags[f"exog_{i}"]].iloc[0])
            
        mean_regressors = np.stack(mean_regressors)
        if self.intercept:
            mean_regressors = np.hstack((np.ones((self.N, 1)), mean_regressors))
            
        bp_resids = sigma_diag_mat @ ybar_vec - sigma_diag_mat @ mean_regressors @ self.beta_mat.mean(axis = 0)
        
        return bp_resids
    
    def swamy_pvalue(self):
        XtX_sigma2_sum = np.einsum('ijk,ijl->kl', self.x_mats / self.sigma2_arr[:, None, None], self.x_mats)
        Xty_sigma2_sum = np.einsum('ijk,ij->k', self.x_mats / self.sigma2_arr[:, None, None], self.y_stack)
        beta_star = np.linalg.inv(XtX_sigma2_sum) @ Xty_sigma2_sum
        swamy_f_stat = 0
        for i in range(self.N):
            swamy_f_stat += (self.beta_mat[i] - beta_star).T @ self.x_mats[i].T @ self.x_mats[i] @ (self.beta_mat[i] - beta_star) / self.sigma2_arr[i]
            
        df = (self.N - 1) * self.k
        
        p_value = 1 - stats.chi2.cdf(swamy_f_stat, df)   # Swamy's test statistic is chi-square under H0: no heterogeneity, not F despite the name
        return swamy_f_stat, p_value
    
def vcm_inference(vcm_model, vcm_data, index_by, panel_id, intercept=True):
    # model_params = vcm_model.beta_mat[panel_id]
    # model_params = vcm_model.beta_gls
    model_params = vcm_model.blue_beta_list[panel_id-1]
    model_ftrs = vcm_model.regressors_with_lags[f'exog_{panel_id}']
    panel_data = vcm_data.loc[vcm_data[index_by] == panel_id]
    
    if intercept:
        vcm_data_matrix = panel_data[model_ftrs].to_numpy()
        vcm_data_matrix = np.hstack((np.ones((vcm_model.T, 1)), vcm_data_matrix))
    else:
        vcm_data_matrix = panel_data[model_ftrs].to_numpy()
        
        
    
    return vcm_data_matrix @ model_params