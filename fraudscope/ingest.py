"""Load tabular inputs and record cleaning counts on DataFrame.attrs."""

from pathlib import Path

import pandas as pd


def load(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load CSV/XLSX, trim strings, coerce fully numeric columns and record NaNs.

    attrs['cleaning'] contains per-column missing counts and source row count.
    Rows are retained, so flags still refer to the input row positions.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xlsm"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Expected a CSV or XLSX file.")
    df.columns = df.columns.astype(str).str.strip()
    if columns is not None:
        df = df[columns].copy()
    for col in df:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            trimmed = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            converted = pd.to_numeric(trimmed, errors="coerce")
            df[col] = converted if converted.notna().sum() == trimmed.notna().sum() else trimmed
    df.attrs["cleaning"] = {"rows": len(df), "dropped_rows": 0,
                            "nan_counts": df.isna().sum().to_dict()}
    return df
