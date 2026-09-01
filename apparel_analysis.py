# ============================================================
# APPAREL & TEXTILES STRATEGIC BUSINESS ANALYSIS
# ============================================================
# End-to-End Data Analytics Project
#
# Workflow:
# Data Loading → Cleaning → EDA → KPI Analysis
# → Visualization → Forecasting → Business Insights
# → Strategic Recommendations
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

# Local dataset file
DATA_FILE = "simulated_apparel_integrated_dataset.csv"

# Public dataset/reference link
DATASET_REFERENCE = (
    "https://www.kaggle.com/search?q=clothing%20sales%20dataset"
)

# Output folders
OUTPUT_DIR = "outputs"
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# 3. PROJECT INFORMATION
# ============================================================

print("=" * 70)
print("APPAREL & TEXTILES STRATEGIC BUSINESS ANALYSIS")
print("=" * 70)

print("\nDataset Reference:")
print(DATASET_REFERENCE)

print("\nProject Objective:")
print(
    "Analyze apparel sales data, identify business trends, "
    "generate KPIs, visualize performance and forecast future revenue."
)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("[1] LOADING DATASET")
print("=" * 70)

if not os.path.exists(DATA_FILE):

    print("\nERROR: Dataset file not found!")
    print(f"Expected file: {DATA_FILE}")
    print("\nMake sure the CSV file is in the same folder as this script.")
    print(f"\nDataset Reference: {DATASET_REFERENCE}")

    exit()

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 5. INITIAL DATA INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("[2] INITIAL DATA INSPECTION")
print("=" * 70)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())


# ============================================================
# 6. DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("[3] DATA CLEANING")
print("=" * 70)


# Remove duplicate rows
duplicates_before = df.duplicated().sum()

df = df.drop_duplicates()

print(
    f"\nDuplicate rows removed: {duplicates_before}"
)


# Convert Month column into datetime
df["Month"] = pd.to_datetime(
    df["Month"],
    errors="coerce"
)


# Remove invalid dates
invalid_dates = df["Month"].isnull().sum()

df = df.dropna(
    subset=["Month"]
)

print(
    f"Invalid date records removed: {invalid_dates}"
)


# Numeric columns
numeric_columns = [
    "Revenue",
    "Units",
    "Discount",
    "Margin",
    "Profit"
]


# Convert numeric columns
for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Handle missing numeric values
for column in numeric_columns:

    missing_count = df[column].isnull().sum()

    if missing_count > 0:

        df[column] = df[column].fillna(
            df[column].median()
        )


# Handle categorical missing values
df["Category"] = df["Category"].fillna(
    "Unknown"
)

df["Region"] = df["Region"].fillna(
    "Unknown"
)


# Remove impossible values
df = df[df["Revenue"] >= 0]
df = df[df["Units"] >= 0]

df = df[
    (df["Discount"] >= 0) &
    (df["Discount"] <= 1)
]

df = df[
    (df["Margin"] >= 0) &
    (df["Margin"] <= 1)
]


# Recalculate profit
df["Profit"] = (
    df["Revenue"] *
    df["Margin"]
)


# Sort data
df = df.sort_values(
    "Month"
).reset_index(drop=True)


print("\nData cleaning completed.")

print(
    f"Final number of records: {len(df):,}"
)


# Save cleaned dataset
cleaned_file = os.path.join(
    OUTPUT_DIR,
    "cleaned_apparel_dataset.csv"
)

df.to_csv(
    cleaned_file,
    index=False
)

print(
    f"Cleaned dataset saved to: {cleaned_file}"
)


# ============================================================
# 7. KEY PERFORMANCE INDICATORS
# ============================================================

print("\n" + "=" * 70)
print("[4] KEY PERFORMANCE INDICATORS")
print("=" * 70)


total_revenue = df["Revenue"].sum()

total_profit = df["Profit"].sum()

total_units = df["Units"].sum()

average_margin = (
    total_profit /
    total_revenue
)

average_discount = (
    df["Discount"].mean()
)


print("\nBusiness KPI Summary")
print("-" * 70)

print(
    f"Total Revenue   : ₹{total_revenue:,.2f}"
)

print(
    f"Total Profit    : ₹{total_profit:,.2f}"
)

print(
    f"Total Units     : {total_units:,.0f}"
)

print(
    f"Average Margin  : {average_margin * 100:.2f}%"
)

print(
    f"Average Discount: {average_discount * 100:.2f}%"
)


# KPI DataFrame
kpi_summary = pd.DataFrame({

    "KPI": [
        "Total Revenue",
        "Total Profit",
        "Total Units Sold",
        "Average Margin",
        "Average Discount"
    ],

    "Value": [
        total_revenue,
        total_profit,
        total_units,
        average_margin,
        average_discount
    ]

})


kpi_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "kpi_summary.csv"
    ),
    index=False
)


# ============================================================
# 8. PRODUCT CATEGORY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("[5] PRODUCT CATEGORY ANALYSIS")
print("=" * 70)


category_analysis = (

    df.groupby("Category")

    .agg(

        Revenue=("Revenue", "sum"),

        Profit=("Profit", "sum"),

        Units=("Units", "sum"),

        Average_Margin=("Margin", "mean"),

        Average_Discount=("Discount", "mean")

    )

    .sort_values(
        "Revenue",
        ascending=False
    )

)


print("\nCategory Performance:")
print(category_analysis)


category_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "category_analysis.csv"
    )
)


best_category = (
    category_analysis.index[0]
)

worst_category = (
    category_analysis.index[-1]
)


print(
    f"\nTop Performing Category: {best_category}"
)

print(
    f"Lowest Performing Category: {worst_category}"
)


# ============================================================
# 9. REGIONAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("[6] REGIONAL ANALYSIS")
print("=" * 70)


region_analysis = (

    df.groupby("Region")

    .agg(

        Revenue=("Revenue", "sum"),

        Profit=("Profit", "sum"),

        Units=("Units", "sum"),

        Average_Margin=("Margin", "mean")

    )

    .sort_values(
        "Revenue",
        ascending=False
    )

)


print("\nRegional Performance:")
print(region_analysis)


region_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "regional_analysis.csv"
    )
)


best_region = (
    region_analysis.index[0]
)

weakest_region = (
    region_analysis.index[-1]
)


print(
    f"\nTop Region: {best_region}"
)

print(
    f"Weakest Region: {weakest_region}"
)


# ============================================================
# 10. MONTHLY SALES ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("[7] MONTHLY SALES ANALYSIS")
print("=" * 70)


monthly_sales = (

    df.groupby("Month")

    .agg(

        Revenue=("Revenue", "sum"),

        Profit=("Profit", "sum"),

        Units=("Units", "sum")

    )

    .reset_index()

)


# Monthly growth
monthly_sales["Revenue_Growth_%"] = (

    monthly_sales["Revenue"]

    .pct_change()

    .fillna(0)

    * 100

)


print("\nMonthly Sales:")
print(monthly_sales.tail(10))


monthly_sales.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "monthly_sales_analysis.csv"
    ),
    index=False
)


# ============================================================
# 11. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("[8] DESCRIPTIVE STATISTICS")
print("=" * 70)


statistics = (

    df[
        [
            "Revenue",
            "Units",
            "Discount",
            "Margin",
            "Profit"
        ]
    ]

    .describe()

)


print(statistics)


statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "descriptive_statistics.csv"
    )
)


# ============================================================
# 12. DISCOUNT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("[9] DISCOUNT ANALYSIS")
print("=" * 70)


discount_correlation = (

    df["Discount"]

    .corr(
        df["Revenue"]
    )

)


print(
    "\nDiscount vs Revenue Correlation:"
)

print(
    f"{discount_correlation:.3f}"
)


print(
    "\nInterpretation:"
)

print(
    "Discount correlation should be evaluated "
    "together with profit and margin."
)


# ============================================================
# 13. CHART 1 – MONTHLY REVENUE TREND
# ============================================================

print("\n" + "=" * 70)
print("[10] GENERATING CHARTS")
print("=" * 70)


plt.figure(
    figsize=(10, 6)
)


plt.plot(

    monthly_sales["Month"],

    monthly_sales["Revenue"] / 1e6,

    marker="o",

    linewidth=2

)


plt.title(
    "Monthly Revenue Trend"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Revenue (₹ Million)"
)

plt.grid(
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        CHART_DIR,
        "monthly_revenue_trend.png"
    ),

    dpi=300

)


plt.close()


# ============================================================
# 14. CHART 2 – CATEGORY REVENUE
# ============================================================

category_revenue = (

    df.groupby("Category")["Revenue"]

    .sum()

    .sort_values()

)


plt.figure(
    figsize=(10, 6)
)


category_revenue.plot(
    kind="barh"
)


plt.title(
    "Revenue by Product Category"
)

plt.xlabel(
    "Revenue (₹)"
)

plt.ylabel(
    "Product Category"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        CHART_DIR,
        "category_revenue.png"
    ),

    dpi=300

)


plt.close()


# ============================================================
# 15. CHART 3 – REGIONAL REVENUE
# ============================================================

region_revenue = (

    df.groupby("Region")["Revenue"]

    .sum()

    .sort_values()

)


plt.figure(
    figsize=(10, 6)
)


region_revenue.plot(
    kind="barh"
)


plt.title(
    "Revenue by Region"
)

plt.xlabel(
    "Revenue (₹)"
)

plt.ylabel(
    "Region"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        CHART_DIR,
        "regional_revenue.png"
    ),

    dpi=300

)


plt.close()


# ============================================================
# 16. CHART 4 – DISCOUNT VS REVENUE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.scatter(

    df["Discount"] * 100,

    df["Revenue"],

    alpha=0.25

)


# Trend line
x = (
    df["Discount"].values *
    100
)

y = (
    df["Revenue"].values
)


coefficients = np.polyfit(
    x,
    y,
    1
)


trend_x = np.linspace(

    x.min(),

    x.max(),

    100

)


trend_y = np.polyval(

    coefficients,

    trend_x

)


plt.plot(

    trend_x,

    trend_y,

    linewidth=2

)


plt.title(
    "Discount Level vs Revenue"
)

plt.xlabel(
    "Discount (%)"
)

plt.ylabel(
    "Revenue (₹)"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        CHART_DIR,
        "discount_vs_revenue.png"
    ),

    dpi=300

)


plt.close()


# ============================================================
# 17. SIX-MONTH SALES FORECAST
# ============================================================

print("\n" + "=" * 70)
print("[11] SIX-MONTH SALES FORECAST")
print("=" * 70)


forecast_data = monthly_sales.copy()


# Time index
forecast_data["Time_Index"] = np.arange(
    len(forecast_data)
)


# Independent variable
X = forecast_data[
    ["Time_Index"]
]


# Target variable
y = forecast_data[
    "Revenue"
]


# Linear Regression model
model = LinearRegression()


model.fit(
    X,
    y
)


# Future time indexes
future_indices = np.arange(

    len(forecast_data),

    len(forecast_data) + 6

).reshape(
    -1,
    1
)


# Generate forecast
forecast_values = model.predict(
    future_indices
)


# Future dates
future_dates = pd.date_range(

    start=forecast_data["Month"].max()
    + pd.offsets.MonthBegin(1),

    periods=6,

    freq="MS"

)


forecast_df = pd.DataFrame({

    "Month": future_dates,

    "Forecast_Revenue": forecast_values

})


print("\nForecast Results:")
print(forecast_df)


forecast_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "six_month_forecast.csv"
    ),

    index=False

)


# ============================================================
# 18. CHART 5 – FORECAST
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    forecast_data["Month"],

    forecast_data["Revenue"] / 1e6,

    marker="o",

    label="Historical Revenue"

)


plt.plot(

    future_dates,

    forecast_values / 1e6,

    marker="o",

    linestyle="--",

    label="Six-Month Forecast"

)


plt.title(
    "Six-Month Revenue Forecast"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Revenue (₹ Million)"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        CHART_DIR,
        "six_month_forecast.png"
    ),

    dpi=300

)


plt.close()


# ============================================================
# 19. BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("[12] BUSINESS INSIGHTS")
print("=" * 70)


insights = [

    f"{best_category} is the highest revenue-generating "
    "product category.",

    f"{best_region} is the strongest performing region.",

    f"{weakest_region} has comparatively lower revenue "
    "and should receive additional market attention.",

    "Revenue demonstrates an overall positive trend "
    "with seasonal variation.",

    "Discounts should be evaluated together with "
    "profitability rather than revenue alone.",

    "Sales forecasting can support production, "
    "inventory and procurement planning."

]


for number, insight in enumerate(
    insights,
    start=1
):

    print(
        f"{number}. {insight}"
    )


# Save insights
insights_df = pd.DataFrame({

    "Business_Insights": insights

})


insights_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "business_insights.csv"
    ),

    index=False

)


# ============================================================
# 20. STRATEGIC RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("[13] STRATEGIC RECOMMENDATIONS")
print("=" * 70)


recommendations = [

    "Adopt demand-based production planning.",

    "Build inventory before high-demand seasonal periods.",

    "Use targeted discounts instead of blanket discounts.",

    "Develop region-specific product assortments.",

    "Review actual sales against forecasts every month.",

    "Monitor revenue together with profit and margin.",

    "Use dashboards for continuous KPI monitoring.",

    "Track forecast accuracy and improve the forecasting model."

]


for number, recommendation in enumerate(

    recommendations,

    start=1

):

    print(
        f"{number}. {recommendation}"
    )


# Save recommendations
recommendation_df = pd.DataFrame({

    "Strategic_Recommendation":
    recommendations

})


recommendation_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "strategic_recommendations.csv"
    ),

    index=False

)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nFINAL BUSINESS SUMMARY")
print("-" * 70)


print(
    f"Total Revenue  : ₹{total_revenue:,.2f}"
)

print(
    f"Total Profit   : ₹{total_profit:,.2f}"
)

print(
    f"Total Units    : {total_units:,.0f}"
)

print(
    f"Average Margin : {average_margin * 100:.2f}%"
)

print(
    f"Average Discount: {average_discount * 100:.2f}%"
)

print(
    f"Top Category   : {best_category}"
)

print(
    f"Top Region     : {best_region}"
)


# ============================================================
# 22. OUTPUT FILES
# ============================================================

print("\nGenerated Output Files:")
print("-" * 70)


output_files = [

    "cleaned_apparel_dataset.csv",

    "kpi_summary.csv",

    "category_analysis.csv",

    "regional_analysis.csv",

    "monthly_sales_analysis.csv",

    "descriptive_statistics.csv",

    "six_month_forecast.csv",

    "business_insights.csv",

    "strategic_recommendations.csv"

]


for file in output_files:

    print(
        f"outputs/{file}"
    )


print("\nGenerated Charts:")
print("-" * 70)


chart_files = [

    "monthly_revenue_trend.png",

    "category_revenue.png",

    "regional_revenue.png",

    "discount_vs_revenue.png",

    "six_month_forecast.png"

]


for file in chart_files:

    print(
        f"outputs/charts/{file}"
    )


# ============================================================
# END OF PROJECT
# ============================================================
