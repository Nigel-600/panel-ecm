from typing import List
import pandas as pd

def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    feature_cols: List[str],
    target_col: str,
    n_lags: int,
    drop_nans: bool = False
) -> pd.DataFrame:
    """
    Adds lag features for a target column grouped by a specified column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_col : str
        Column name to group by (e.g., 'Store').
    feature_cols : List[str]
        Additional feature columns to also create lags for.
    target_col : str
        Column name to create lag features for (e.g., 'Weekly_Sales').
    n_lags : int
        Number of lag features to create.
    drop_nans : bool
        Whether to drop rows containing NaNs introduced by lagging.

    Returns
    -------
    pd.DataFrame
        Dataframe with new lag features added.
    """
    df_with_lag = df.copy()
    df_with_lag = df_with_lag.sort_values(by=[group_col, "DateInt"])  # ensure proper ordering

    for lag in range(1, n_lags + 1):
        df_with_lag[f"{target_col}_L{lag}"] = (
            df_with_lag.groupby(group_col)[target_col].shift(lag)
        )
        if feature_cols is None:
            continue
        else:
            for ftr in feature_cols:
                df_with_lag[f"{ftr}_L{lag}"] = (
                    df_with_lag.groupby(group_col)[ftr].shift(lag)
                )

    return df_with_lag.dropna() if drop_nans else df_with_lag