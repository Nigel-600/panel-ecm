import numpy as np
import pandas as pd
from scipy import stats
import scipy.linalg as la
import matplotlib.pyplot as plt

from panelmodels import VariableCoefficientModel, vcm_inference, add_lag_features

def main():
    wm45_df = pd.read_csv("data/wm45_df.csv")
    
    index_id = "Store"
    vcm_features = {}
    vcm_response = {"y" : ["Weekly_Sales"]}
    mdl_intercept=True
    for i in wm45_df[index_id].unique():
        vcm_features[f"exog_{i}"] = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

    n_lags=0
    walmart_vcm = VariableCoefficientModel(index_by=index_id)

    _, lm_stats = walmart_vcm.fit(
        panel_df = wm45_df,
        regressors = vcm_features,
        response=vcm_response,
        lags=n_lags,
        model_type="fixed",
        intercept=mdl_intercept, 
        asymptotic=True,
        bayes_estimator=False
    )
    print(lm_stats)
    
    
    
if __name__ == "__main__":
    main()