from __future__ import annotations
import pandas as pd
import numpy as np

try:
    import jdatetime
except ImportError:
    jdatetime = None


def _detect_date_column(df: pd.DataFrame) -> str | None:
    candidates = ["Date", "date", "DATE", "تاریخ", "تاريخ", "Tarikh"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return None


def _detect_price_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Close", "close", "Adj Close", "adj_close",
        "Price", "price", "ClosePrice",
        "قیمت", "قيمت", "قیمت پایانی", "قیمت_پایانی",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return numeric_cols[-1] if len(numeric_cols) else None


def remove_duplicates_from_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df = xls.parse(sheet_name).copy()
    date_col = _detect_date_column(df)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.drop_duplicates()
    if date_col is not None:
        df = df.sort_values(date_col)
    return df.reset_index(drop=True)


def prepare_asset_data_from_cleaned(cleaned_df: pd.DataFrame, asset_name: str) -> pd.Series:
    if cleaned_df is None or not isinstance(cleaned_df, pd.DataFrame) or cleaned_df.empty:
        raise ValueError(f"cleaned_df for asset '{asset_name}' is empty or invalid.")

    date_col = _detect_date_column(cleaned_df)
    price_col = _detect_price_column(cleaned_df)

    if date_col is None or price_col is None:
        raise ValueError(
            f"Could not detect date/price columns for asset '{asset_name}'."
        )

    df = cleaned_df[[date_col, price_col]].copy()
    df = df.dropna(subset=[price_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    returns = df[price_col].pct_change()
    series_name = f"بازده_{asset_name}"
    returns.name = series_name
    return returns


def calculate_statistics(returns_df: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(index=returns_df.columns)
    stats["mean"] = returns_df.mean()
    stats["std"] = returns_df.std()
    stats["min"] = returns_df.min()
    stats["max"] = returns_df.max()
    stats["skew"] = returns_df.skew()
    stats["kurtosis"] = returns_df.kurtosis()
    stats["median"] = returns_df.median()
    stats["count"] = returns_df.count()
    return stats


def calculate_risk_metrics(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
):
    mean_daily = returns_df.mean()
    std_daily = returns_df.std()

    ann_return = mean_daily * periods_per_year
    ann_vol = std_daily * np.sqrt(periods_per_year)

    sharpe = (ann_return - risk_free_rate) / ann_vol.replace(0, np.nan)

    metrics = pd.DataFrame({
        "mean_daily_return": mean_daily,
        "std_daily_return": std_daily,
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
    })

    return metrics, std_daily


def calculate_portfolio_VaR(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    from scipy.stats import norm

    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != returns_df.shape[1]:
        raise ValueError("weights length must match number of columns in returns_df.")

    mean_vector = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    port_mean = float(weights @ mean_vector)
    port_std = float(np.sqrt(weights @ cov_matrix @ weights))

    z = norm.ppf(1 - confidence_level)
    var = port_mean + z * port_std
    return var


def persian_text(text: str) -> str:
    return text


def to_jalali(date) -> str:
    if pd.isna(date):
        return ""
    if jdatetime is None:
        return pd.to_datetime(date).strftime("%Y-%m-%d")
    dt = pd.to_datetime(date).to_pydatetime()
    j = jdatetime.date.fromgregorian(day=dt.day, month=dt.month, year=dt.year)
    return f"{j.year:04d}-{j.month:02d}-{j.day:02d}"
