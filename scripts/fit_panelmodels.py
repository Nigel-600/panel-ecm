import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from panelmodels import VariableCoefficientModel, swamy_prediction, sur_prediction


def get_within_sample_predictions(vcm_model, prediction_fn, index_by, response_col, intercept=True):
    """Runs prediction_fn for every individual in the fitted model and stacks
    actual vs. predicted values into a single DataFrame."""
    vcm_data = vcm_model.panel_df_with_lag
    original_ids = vcm_model.id_encoder.classes_   # original (pre-encoding) ids, in encoder order

    records = []
    for orig_id in original_ids:
        y_pred = prediction_fn(
            vcm_model=vcm_model,
            vcm_data=vcm_data,
            index_by=index_by,
            panel_id=orig_id,
            intercept=intercept,
        )
        y_actual = vcm_data.loc[vcm_data[index_by] == orig_id, response_col].to_numpy()

        records.append(pd.DataFrame({
            index_by: orig_id,
            "actual": y_actual,
            "predicted": y_pred,
        }))

    return pd.concat(records, ignore_index=True)


def plot_within_sample_predictions(pred_df, index_by, model_name, n_series_to_plot=4):
    """Two-panel diagnostic: pooled actual-vs-predicted scatter, and a handful
    of individual time series overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(pred_df["actual"], pred_df["predicted"], s=8, alpha=0.4)
    lims = [
        min(pred_df["actual"].min(), pred_df["predicted"].min()),
        max(pred_df["actual"].max(), pred_df["predicted"].max()),
    ]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y = ŷ")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name}: actual vs. predicted (all {index_by}s, in-sample)")
    ax.legend()

    ss_res = np.sum((pred_df["actual"] - pred_df["predicted"]) ** 2)
    ss_tot = np.sum((pred_df["actual"] - pred_df["actual"].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, va="top")

    # ---- Panel 2: a few individual series over time (actual vs predicted) ----
    ax = axes[1]
    sample_ids = pred_df[index_by].unique()[:n_series_to_plot]
    for sid in sample_ids:
        sub = pred_df.loc[pred_df[index_by] == sid].reset_index(drop=True)
        ax.plot(sub.index, sub["actual"], alpha=0.6, label=f"{sid} actual")
        ax.plot(sub.index, sub["predicted"], linestyle="--", alpha=0.8, label=f"{sid} pred")
    ax.set_xlabel("Time index (within series)")
    ax.set_ylabel("Response")
    ax.set_title(f"{model_name}: sample of {n_series_to_plot} series")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    return fig


def plot_single_series_prediction(pred_df, index_by, panel_id, model_name):
    """Actual vs. predicted over time for one individual series."""
    sub = pred_df.loc[pred_df[index_by] == panel_id].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sub.index, sub["actual"], label="Actual", linewidth=1.5)
    ax.plot(sub.index, sub["predicted"], label="Predicted", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Time index (within series)")
    ax.set_ylabel("Response")
    ax.set_title(f"{model_name}: {index_by} = {panel_id}, in-sample fit")
    ax.legend()

    resid = sub["actual"] - sub["predicted"]
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((sub["actual"] - sub["actual"].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, va="top")

    fig.tight_layout()
    return fig


def main():
    wm45_df = pd.read_csv("data/wm45_df.csv")

    index_id = "Store"
    vcm_features = {}
    vcm_response = {"y": ["Weekly_Sales"]}
    mdl_intercept = True
    for i in wm45_df[index_id].unique():
        vcm_features[f"exog_{i}"] = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

    n_lags = 3


    walmart_swamy = VariableCoefficientModel(index_by=index_id)
    walmart_swamy.fit(
        panel_df=wm45_df,
        regressors=vcm_features,
        response=vcm_response,
        lags=n_lags,
        model_type="random",
        intercept=mdl_intercept,
        asymptotic=True,
    )

    swamy_stat, swamy_p = walmart_swamy.swamy_pvalue()
    print("Swamy homogeneity test statistic:", swamy_stat)
    print("Swamy homogeneity p-value:", swamy_p)

    bp_stats = walmart_swamy.gen_breusch_pagan()
    print("Breusch-Pagan (Swamy fit):", bp_stats)

    swamy_pred_df = get_within_sample_predictions(
        vcm_model=walmart_swamy,
        prediction_fn=swamy_prediction,
        index_by=index_id,
        response_col=vcm_response["y"][0],
        intercept=mdl_intercept,
    )

    walmart_sur = VariableCoefficientModel(index_by=index_id)
    walmart_sur.fit(
        panel_df=wm45_df,
        regressors=vcm_features,
        response=vcm_response,
        lags=n_lags,
        model_type="fixed",
        intercept=mdl_intercept,
        asymptotic=True,
    )

    sur_diagnostics = walmart_sur.sur_stats()
    print("SUR diagnostics:", sur_diagnostics)

    sur_pred_df = get_within_sample_predictions(
        vcm_model=walmart_sur,
        prediction_fn=sur_prediction,
        index_by=index_id,
        response_col=vcm_response["y"][0],
        intercept=mdl_intercept,
    )

    example_store = wm45_df[index_id].unique()[0]

    plot_single_series_prediction(swamy_pred_df, index_id, panel_id=example_store, model_name="Swamy (random coefficients)")
    plot_single_series_prediction(sur_pred_df, index_id, panel_id=example_store, model_name="SUR (fixed coefficients)")

    plt.show()


if __name__ == "__main__":
    main()