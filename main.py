from enum import Enum
import json
import sys
from typing import Any, Dict, Literal, Union

from dataprep.eda.configs import Config  # type: ignore
from dataprep.eda.create_report.formatter import format_report  # type: ignore
import numpy as np
import pandas as pd


def is_near_zero_variance(series: pd.Series, threshold=0.01) -> bool:
    """Check if a series has near zero variance."""
    if series.std() == 0 or series.mean() == 0:
        return False
    return (series.std() / series.mean()) < threshold


DType = Union[Literal["Numeric"], Literal["Categorical"], Literal["Other"], Literal["Text"]]


def categorize_dtype(df: pd.DataFrame, column: str) -> DType:
    """Categorize data type as 'Numeric', 'Categorical', or 'Text'."""
    # Identify as categorical if high uniqueness ratio (potential row identifier)
    uniqueness_ratio = df[column].nunique() / len(df)
    if uniqueness_ratio > 0.95:  # Adjust this threshold as needed
        return 'Categorical'

    if pd.api.types.is_numeric_dtype(df[column]):
        return 'Numeric'
    elif pd.api.types.is_object_dtype(df[column]):
        median_length = df[column].dropna().astype(str).map(len).median()
        return 'Categorical' if median_length < 50 else 'Text'
    else:
        return 'Other'


def vj_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame()

    summary['Var Name'] = df.columns
    summary['Data Type'] = summary['Var Name'].apply(lambda v: categorize_dtype(df, v))

    summary['Unique'] = df.nunique().values
    summary['Missing'] = df.isnull().sum().values

    # Compute statistics only for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_stats = df[numeric_cols].agg(['min', 'mean', 'median', 'max', 'std'])

    # Merge statistics with the summary dataframe
    for stat in numeric_stats.index:
        summary[stat.capitalize()] = summary['Var Name'].map(numeric_stats.loc[stat])

    # Zero variance
    summary['ZV'] = summary['Var Name'].map(df[numeric_cols].std() == 0)
    summary['ZV'] = summary['ZV'].map({True: 'Yes', False: ''})

    # Near zero variance
    summary['NZV'] = df[numeric_cols].apply(is_near_zero_variance)
    summary['NZV'] = summary['NZV'].map({True: 'Yes', False: ''})

    # Replace NaNs with blanks
    summary = summary.fillna('')

    # Limit numeric values to two decimal points
    summary = summary.applymap(lambda x: f"{x:.2f}" if isinstance(x, (float, np.float64)) else x)

    # Adjust index to start from 1
    summary.index = range(1, len(summary) + 1)
    summary.index.name = 'Index'

    return summary


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series, pd.Index)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('index')
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            return f"<<{type(obj)}>>"


def clean_dict(d):
    """
    Recursively clean a dictionary to make it serializable to JSON.
    Removes any keys that are not of type str, int, float, bool, or None.

    Args:
        d (dict): The dictionary to clean.

    Returns:
        dict: The cleaned dictionary.
    """
    def is_json_serializable(v):
        return isinstance(v, (Enum, str, int, float, bool, type(None)))

    def xformk(k):
        if isinstance(k, Enum):
            return str(k)
        else:
            return k

    if not isinstance(d, dict):
        raise ValueError("Input should be a dictionary.")

    return {xformk(k): clean_dict(v) if isinstance(v, dict) else v for k, v in d.items() if is_json_serializable(k)}


def summarize_dataset(path: str) -> Dict[str, Any]:
    if path.endswith(".csv"):
        df = pd.read_csv(path, encoding_errors="ignore")
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise Exception("unrecognized file")

    cfg = Config.from_dict(None, None)
    report = format_report(df, cfg, "basic")

    vj_table = vj_summary(df)
    report["vj_table"] = vj_table
    return report


def main(path):
    res = summarize_dataset(path)

    cleaned = clean_dict(res)

    json.dump(cleaned, sys.stdout, cls=NumpyEncoder)


if __name__ == "__main__":
    main(sys.argv[1])
