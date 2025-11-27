# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from portfolio_utils import (
    remove_duplicates_from_sheet,
    prepare_asset_data_from_cleaned,
    calculate_statistics,
    calculate_risk_metrics,
    calculate_portfolio_VaR,
    persian_text,
    to_jalali,
)


def main():
    file_path = "data/portfolio_data.xlsx"
    xls = pd.ExcelFile(file_path)

    sheet_names = xls.sheet_names
    print("تعداد ردیف‌ها قبل از پاکسازی:")
    for sheet in sheet_names:
        df_unclean = xls.parse(sheet)
        print(f"  {sheet}: {len(df_unclean)}")

    cleaned_data = {}
    for sheet in sheet_names:
        try:
            cleaned_df = remove_duplicates_from_sheet(xls, sheet)
            cleaned_data[sheet] = cleaned_df
        except Exception as e:
            cleaned_data[sheet] = f"خطا در پردازش: {e}"

    print("\nتعداد ردیف‌ها بعد از پاکسازی:")
    for sheet, df in cleaned_data.items():
        if isinstance(df, pd.DataFrame):
            print(f"  {sheet}: {len(df)}")
        else:
            print(f"  {sheet}: {df}")

    assets = ["فولاد", "فزر", "سغرب", "پترول", "اعتماد"]

    returns_series_list = [
        prepare_asset_data_from_cleaned(cleaned_data[asset], asset)
        for asset in assets
    ]
    returns_df = pd.concat(returns_series_list, axis=1).dropna()

    column_translation = {
        "بازده_فولاد": "Return_Steel",
        "بازده_فزر": "Return_Gold",
        "بازده_سغرب": "Return_S.Gharb",
        "بازده_پترول": "Return_Petrol",
        "بازده_اعتماد": "Return_Etemad",
    }
    returns_df = returns_df.rename(columns=column_translation)
    returns_df.index.name = "Date"

    print("\nDescriptive Statistics:")
    statistics_summary = calculate_statistics(returns_df)
    print(statistics_summary.round(4))

    window_size = 252
    rolling_corr = returns_df.rolling(window=window_size).corr().dropna()

    try:
        rolling_corr_pair = rolling_corr.loc[(slice(None), "Return_Steel"), "Return_S.Gharb"]
        rolling_corr_pair.index = rolling_corr_pair.index.droplevel(1)
    except KeyError:
        print("\nWarning: Could not compute rolling correlation for Steel & S.Gharb.")
        rolling_corr_pair = None

    if rolling_corr_pair is not None and not rolling_corr_pair.empty:
        plt.figure(figsize=(10, 4))
        rolling_corr_pair.plot(title="Rolling Correlation: Steel & S.Gharb")
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylim(-1, 1)
        plt.tight_layout()
        plt.show()

    risk_metrics, std_devs = calculate_risk_metrics(returns_df)
    print("\nRisk Metrics:")
    print(risk_metrics)

    weights = np.array([0.4, 0.3, 0.3, 0.0, 0.0])

    cov_matrix = returns_df.cov().values
    portfolio_std = float(np.sqrt(weights.T @ cov_matrix @ weights))

    confidence_level = 0.95
    z_score = norm.ppf(1 - confidence_level)
    VaR_values = returns_df.mean() + z_score * std_devs
    portfolio_VaR = float(weights @ returns_df.mean().values + z_score * portfolio_std)

    print("\nPortfolio Risk Metrics:")
    print({
        "Standard_Deviation_Assets": std_devs.to_dict(),
        "Portfolio_Standard_Deviation": portfolio_std,
        "Value_at_Risk_Assets": VaR_values.to_dict(),
        "Portfolio_Value_at_Risk": portfolio_VaR,
    })

    weights_original = np.array([0.4, 0.3, 0.3, 0.0, 0.0])
    weights_new = np.array([0.3, 0.5, 0.2, 0.0, 0.0])

    portfolio_VaR_original = calculate_portfolio_VaR(returns_df, weights_original)
    portfolio_VaR_new = calculate_portfolio_VaR(returns_df, weights_new)

    print("\nPortfolio Value at Risk (Original Weights):", portfolio_VaR_original)
    print("Portfolio Value at Risk (Increased Gold Weight):", portfolio_VaR_new)

    VaR_values_simple = returns_df.apply(lambda x: norm.ppf(0.05) * x.std())

    plt.figure(figsize=(8, 4))
    VaR_values_simple.plot(
        kind="bar",
        title="Value at Risk (VaR 95%) for Assets",
        color="red",
        edgecolor="black",
    )
    plt.ylabel("VaR")
    plt.xlabel("Assets")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Original Weights", "Increased Gold Weight"],
        [portfolio_VaR_original, portfolio_VaR_new],
        color=["blue", "orange"],
    )
    plt.title("Comparison of Portfolio Value at Risk")
    plt.ylabel("VaR")
    plt.tight_layout()
    plt.show()

    returns_df_clean = returns_df.dropna()
    weekly_returns = returns_df_clean.resample("W").sum()
    cumulative_returns = (1 + weekly_returns).cumprod()

    plt.figure(figsize=(12, 6))
    for col in cumulative_returns.columns:
        label = persian_text(col.replace("Return_", ""))
        plt.plot(cumulative_returns.index, cumulative_returns[col], label=label)

    plt.title(persian_text("بازده تجمعی دارایی‌ها (هفتگی)"))
    plt.xlabel(persian_text("تاریخ (شمسی)"))
    plt.ylabel(persian_text("بازده تجمعی"))

    tick_idx = cumulative_returns.index[:: max(1, len(cumulative_returns) // 10)]
    tick_labels = [to_jalali(d) for d in tick_idx]
    plt.xticks(ticks=tick_idx, labels=tick_labels, rotation=45)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    equal_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
    portfolio_returns = returns_df.values @ equal_weights

    plt.figure(figsize=(10, 5))
    plt.hist(portfolio_returns, bins=30, density=True, alpha=0.7, edgecolor="black")
    plt.axvline(
        portfolio_returns.mean(),
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=persian_text("میانگین بازده"),
    )
    plt.title(persian_text("هیستوگرام بازده پرتفوی (وزن برابر)"))
    plt.xlabel(persian_text("بازده"))
    plt.ylabel(persian_text("چگالی"))
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
