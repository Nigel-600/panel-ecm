import duckdb
import polars as pl
import polars.selectors as cs
import os
import logging

import numpy as np
import pandas as pd

import argparse

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure root logging for the script.

    Parameters
    ----------
    level : logging level name, e.g. "DEBUG", "INFO", "WARNING"
    log_file : optional path to also write logs to a file
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def csv_to_parquet_lazyframe(csv_path: str, pq_path: str, force: bool = False, compression: str = "zstd") -> pl.LazyFrame:
    """
    Convert a CSV to parquet (if the parquet doesn't already exist, or force=True),
    then return a LazyFrame scanning the parquet version.

    Parameters
    ----------
    csv_path : path to the source CSV file
    pq_path : path to write/read the parquet file
    force : if True, re-convert even if the parquet already exists
    compression : parquet compression codec
    """
    if force or not os.path.exists(pq_path):
        logger.info("Converting CSV to parquet: %s -> %s (compression=%s)", csv_path, pq_path, compression)
        pl.scan_csv(csv_path).sink_parquet(pq_path, compression=compression)
        logger.debug("Parquet file written to %s", pq_path)
    else:
        logger.info("Parquet already exists at %s, skipping conversion (force=%s)", pq_path, force)

    logger.debug("Returning LazyFrame scanning %s", pq_path)
    return pl.scan_parquet(pq_path)


def save_pivot(wm45_pl_df: pl.DataFrame, values: str, output_path: str, filename: str) -> None:
    """
    Pivot wm45_pl_df on `values` (with Store as columns) and save the result to CSV.

    Parameters
    ----------
    wm45_pl_df : source polars DataFrame
    values : column to fill pivoted values with
    output_path : directory to write the CSV into
    filename : name of the output CSV file
    """
    logger.info("Pivoting on '%s' -> %s", values, filename)
    pivot_df = wm45_pl_df.pivot(
        values=values,
        index=["Date", "DateInt"],
        on="Store",
    ).to_pandas()

    out_path = os.path.join(output_path, filename)
    pivot_df.to_csv(out_path, index=False)
    logger.info("Wrote %s (shape=%s)", out_path, pivot_df.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walmart aggregate sales per store")
    parser.add_argument("--output_path", type=str, default=r"data", help="Path to output aggregate weekly sales and other regressors' data in csv. Should be a path relative to current directory of code execution.")
    parser.add_argument("--scale", type=int, default=0, help="To scale the weekly sales revenue by the million. 0 to avoid scaling. Will normalise otherwise.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--log-file", type=str, default=None, help="Optional path to write logs to a file, in addition to stdout")
    args = parser.parse_args()

    setup_logging(level=args.log_level, log_file=args.log_file)
    logger.info("Starting Walmart aggregate sales processing")
    logger.debug("Parsed args: %s", args)

    os.makedirs(args.output_path, exist_ok=True)

    wm45_csv_path = os.path.join("data", "walmart-sales-dataset-of-45stores.csv")
    wm45_pq_path = os.path.join(args.output_path, "walmart-sales-dataset-of-45stores.parquet")
    logger.debug("Source CSV: %s", wm45_csv_path)
    logger.debug("Target parquet: %s", wm45_pq_path)
    # The original walmart data is saved as a parquet file. Additional processing is done on the lazyframe obtained from reading this parquet file
    # Due to the simplicity of the data files, the data are saved as csv. Just the processing is done on the lazyframe. Hence the integration of parquet data processing is quite minimal

    # walmart45.csv -> walmart45.parquet in output_path
    wm45_pq = csv_to_parquet_lazyframe(
        csv_path=wm45_csv_path,
        pq_path=wm45_pq_path
    )

    # Execute a query to read lazyframe
    logger.info("Reading parquet via DuckDB")
    query = f"""
        SELECT * 
        FROM '{wm45_pq_path}'
    """
    con = duckdb.connect()
    wm45_lf = con.execute(query).pl().lazy()
    logger.debug("Loaded lazyframe with columns: %s", wm45_lf.collect_schema().names())

    # Processing the lazyframe to add another "DateInt" column.
    # Note that the data are recorded on a weekly basis.
    # Dates are not consecutive dates as what DateInt may reflect,
    # The dates are consecutive only on the week-level

    # Date is mapped to an integer equal to the number of seconds past the UNIX baseline
    logger.info("Deriving DateInt column from Date")
    wm45_lf = wm45_lf.with_columns(
        pl.col("Date").alias("DateInt")
    ).select(
        ["Date", "DateInt"] + [c for c in wm45_lf.collect_schema().names() if c not in ("Date", "DateInt")]
    )
    wm45_lf = wm45_lf.with_columns(
        pl.col("DateInt")
        .str.strptime(pl.Date, "%d-%m-%Y")
        .cast(pl.Int32)  # days since epoch
        .alias("DateInt")
    )

    # Normalise so that DateInt are consecutive integers on the week level
    logger.debug("Normalising DateInt to consecutive weekly integers")
    wm45_lf = wm45_lf.with_columns(
        ((pl.col("DateInt") - pl.col("DateInt").min()) / 7 + 1).alias("DateInt")
        .cast(pl.Int32)
    )

    # Obtain polars DataFrame. Not a pandas dataframe yet at this stage
    # Pivoting is done on this DataFrame
    logger.info("Collecting LazyFrame into DataFrame")
    wm45_pl_df = wm45_lf.collect()
    logger.info("Collected DataFrame with shape %s", wm45_pl_df.shape)

    # Following dataframes are pivotted data
    # After pivotting, saved as pandas DataFrame then saved as a csv in output_path
    # Eases plotting, but not necessary for this library's model fitting, only pass data that looks like wm45_df
    save_pivot(wm45_pl_df, "Weekly_Sales", args.output_path, "wm45_sales_pivot.csv")
    save_pivot(wm45_pl_df, "Temperature", args.output_path, "wm45_temp_pivot.csv")
    save_pivot(wm45_pl_df, "Fuel_Price", args.output_path, "wm45_fp_pivot.csv")
    save_pivot(wm45_pl_df, "CPI", args.output_path, "wm45_cpi_pivot.csv")
    save_pivot(wm45_pl_df, "Unemployment", args.output_path, "wm45_ue_pivot.csv")

    # Finally, wm45_pl_df to a pandas DataFrame as major processing is complete
    logger.info("Converting final DataFrame to pandas")
    wm45_df = wm45_pl_df.to_pandas()
    if args.scale:
        logger.info("Scaling Weekly_Sales by 1,000,000")
        wm45_df["Weekly_Sales"] = wm45_df["Weekly_Sales"] / 1_000_000
    else:
        logger.info("Skipping Weekly_Sales scaling (--scale=0)")


    final_out_path = os.path.join(args.output_path, "wm45_df.csv")
    wm45_df.to_csv(final_out_path, index=False)
    logger.info("Wrote final dataset to %s (shape=%s)", final_out_path, wm45_df.shape)
    logger.info("Processing complete")