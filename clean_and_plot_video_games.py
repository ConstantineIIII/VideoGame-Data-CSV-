from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV from the same folder as this script
csv_path = Path(__file__).parent / "Sample_worldwide_video_games.csv"
df = pd.read_csv(csv_path)

# ---------------------------------------------------------------
# 1. Turn placeholder junk into real NaN
# ---------------------------------------------------------------
df = df.replace(["unknown", "Unknown", "TBD", "many", "N/A", ""], np.nan)

# ---------------------------------------------------------------
# 2. Fix numeric columns (strip $, commas, "M" suffix, words)
# ---------------------------------------------------------------
def to_number(series):
    """Strip $ , and M from a column, then convert to numbers.
    Anything unconvertible becomes NaN."""
    cleaned = (series.astype(str)
                     .str.replace("$", "", regex=False)
                     .str.replace(",", "", regex=False)
                     .str.replace("M", "", regex=False)
                     .str.strip())
    return pd.to_numeric(cleaned, errors="coerce")

# "free" means the price is 0, so handle it before converting
df["Price_USD"] = df["Price_USD"].replace("free", "0")

# "low"/"high" buzz scores: map to numbers on the 0-100 scale
df["Social_Buzz_Score"] = df["Social_Buzz_Score"].replace({"low": "25", "high": "75"})

for col in ["Price_USD", "Review_Count", "Monthly_Active_Users",
            "Units_Sold_Millions", "Revenue_M_USD", "Social_Buzz_Score"]:
    df[col] = to_number(df[col])

# ---------------------------------------------------------------
# 3. Standardize categorical columns
# ---------------------------------------------------------------
df["Genre"] = df["Genre"].str.strip().str.title()   # action -> Action, rpg -> Rpg
df["Genre"] = df["Genre"].replace({"Rpg": "RPG"})   # fix the acronym

platform_map = {
    "ps5": "PS5",
    "XBOX": "Xbox",
    "Xbox Series X": "Xbox",
    "Nintendo Switch": "Switch",
}
df["Platform"] = df["Platform"].str.strip().replace(platform_map)

region_map = {
    "EU": "Europe",
    "APAC": "Asia-Pacific",
}
df["Region"] = df["Region"].str.strip().replace(region_map)

# ---------------------------------------------------------------
# 4. Handle missing values, column by column
# ---------------------------------------------------------------
# Text columns: label as unknown rather than dropping rows
for col in ["Publisher", "Genre", "Platform", "Region", "Target_Audience",
            "Monetization_Model", "Market_Positioning", "Data_Source"]:
    df[col] = df[col].fillna("Unknown")

# Numeric columns: NOT filled. We are plotting, so missing values are
# dropped per plot (dropna) instead of invented. Filling with the median
# would put fake points/bars in the charts.

# Free-text columns: fill with empty string
for col in ["Trend_Tag", "Competitor_Weakness", "Gap_Opportunity", "Notes"]:
    df[col] = df[col].fillna("")

# Dates: convert to real datetime. 5 rows fail (2 blank + 3 unparseable);
# they become NaT and are excluded from the time plot below.
df["Research_Date"] = pd.to_datetime(df["Research_Date"], errors="coerce")

# ---------------------------------------------------------------
# 5. Verify and save
# ---------------------------------------------------------------
print(df.info())
print("\nRemaining missing values:\n", df.isna().sum())

df.to_csv(Path(__file__).parent / "Sample_worldwide_video_games_CLEAN.csv", index=False)
print("\nSaved cleaned file.")

# ---------------------------------------------------------------
# 6. Plots — each on its own figure, NaN dropped per plot
# ---------------------------------------------------------------

# Bar: average revenue by genre (.mean() skips NaN automatically)
plt.figure()
df.groupby("Genre")["Revenue_M_USD"].mean().plot(kind="bar")
plt.title("Average Revenue by Genre")
plt.ylabel("Revenue (M USD)")
plt.tight_layout()

# Scatter: price vs rating (drop rows missing either of these two columns)
plot_df = df.dropna(subset=["Price_USD", "Avg_User_Rating"])
plt.figure()
plt.scatter(plot_df["Price_USD"], plot_df["Avg_User_Rating"])
plt.title("Price vs. User Rating")
plt.xlabel("Price (USD)")
plt.ylabel("Avg User Rating")
plt.tight_layout()

# Histogram: price distribution
plt.figure()
df["Price_USD"].dropna().plot(kind="hist", bins=20)
plt.title("Price Distribution")
plt.xlabel("Price (USD)")
plt.tight_layout()

# Time series: revenue over research date (excludes the 5 NaT rows)
dated = df.dropna(subset=["Research_Date", "Revenue_M_USD"]).sort_values("Research_Date")
plt.figure()
plt.plot(dated["Research_Date"], dated["Revenue_M_USD"], marker="o", linestyle="")
plt.title("Revenue by Research Date")
plt.xlabel("Research Date")
plt.ylabel("Revenue (M USD)")
plt.tight_layout()

plt.show()   # display all four windows; script pauses here until closed
