"""
Game Concept Advisor
Analyzes the cleaned video game market data and recommends:
  - what kind of game to build (genre, platform, audience, monetization)
  - what price to set
  - what time of year to release
Adjust EFFORT_LEVEL below to match the size of your team/budget.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# =================================================================
# SETTING: how much development effort can you afford?
#   "low"    = small team / first game  (simpler genres)
#   "medium" = mid-size team
#   "high"   = large team / big budget  (all genres allowed)
# =================================================================
EFFORT_LEVEL = "low"

# Genres grouped by typical development effort (industry rule of thumb:
# content volume, netcode, and 3D/physics complexity drive cost)
EFFORT_TIERS = {
    "low":    ["Puzzle", "Adventure", "Strategy"],
    "medium": ["Puzzle", "Adventure", "Strategy", "Simulation", "RPG",
               "Racing", "Sports"],
    "high":   ["Puzzle", "Adventure", "Strategy", "Simulation", "RPG",
               "Racing", "Sports", "Action", "Shooter", "Battle Royale"],
}

# ---------------------------------------------------------------
# Load and apply last-mile fixes the cleaning pass missed
# ---------------------------------------------------------------
csv_path = Path(__file__).parent / "Sample_worldwide_video_games_CLEAN.csv"
df = pd.read_csv(csv_path)

# Monetization still has duplicate spellings
df["Monetization_Model"] = df["Monetization_Model"].replace({
    "premium": "Premium",
    "Free to Play": "Free-to-Play",
})

# Ratings outside the 0-10 scale are data errors -> treat as missing
df.loc[(df["Avg_User_Rating"] < 0) | (df["Avg_User_Rating"] > 10),
       "Avg_User_Rating"] = np.nan

df["Research_Date"] = pd.to_datetime(df["Research_Date"], errors="coerce")
df["Month"] = df["Research_Date"].dt.month

# =================================================================
# 1. SCORE THE GENRES
# Composite score = revenue 40% + rating 20% + buzz 20% + openness 20%
# "Openness" rewards genres with FEWER existing titles (less saturated).
# =================================================================
genres = (df[df["Genre"] != "Unknown"]
          .groupby("Genre")
          .agg(titles=("Game_Title", "count"),
               avg_revenue=("Revenue_M_USD", "mean"),
               avg_rating=("Avg_User_Rating", "mean"),
               avg_buzz=("Social_Buzz_Score", "mean")))

def normalize(s):
    """Scale a column to 0-1 so different units can be combined."""
    return (s - s.min()) / (s.max() - s.min())

genres["score"] = (0.4 * normalize(genres["avg_revenue"])
                   + 0.2 * normalize(genres["avg_rating"])
                   + 0.2 * normalize(genres["avg_buzz"])
                   + 0.2 * (1 - normalize(genres["titles"])))  # fewer = better

genres = genres.sort_values("score", ascending=False)

print("=" * 64)
print("GENRE SCOREBOARD (all effort levels)")
print("=" * 64)
print(genres.round(2).to_string())

# Restrict to genres feasible at the chosen effort level
allowed = EFFORT_TIERS[EFFORT_LEVEL]
feasible = genres[genres.index.isin(allowed)]
best_genre = feasible.index[0]

print(f"\nEffort level: {EFFORT_LEVEL} -> feasible genres: {allowed}")
print(f">>> RECOMMENDED GENRE: {best_genre}")

# =================================================================
# 2. PLATFORM, AUDIENCE, MONETIZATION within the recommended genre
# (fall back to the whole dataset when the genre slice is too small)
# =================================================================
def best_option(frame, column, metric="Revenue_M_USD", min_n=3):
    """Highest-average-metric value of `column`, needing >= min_n rows;
    falls back to the full dataset if the slice is too thin."""
    stats = (frame[frame[column] != "Unknown"]
             .groupby(column)[metric].agg(["count", "mean"]))
    stats = stats[stats["count"] >= min_n]
    if stats.empty:                      # genre slice too small -> use all data
        stats = (df[df[column] != "Unknown"]
                 .groupby(column)[metric].agg(["count", "mean"]))
        stats = stats[stats["count"] >= min_n]
    return stats["mean"].idxmax(), stats

genre_df = df[df["Genre"] == best_genre]

platform, plat_stats = best_option(genre_df, "Platform")
audience, aud_stats = best_option(genre_df, "Target_Audience")
monetization, mon_stats = best_option(genre_df, "Monetization_Model")

print("\n" + "=" * 64)
print(f"BEST FIT WITHIN / AROUND '{best_genre}'")
print("=" * 64)
print(f"Platform:     {platform}")
print(f"Audience:     {audience}")
print(f"Monetization: {monetization}")

# =================================================================
# 3. PRICE RECOMMENDATION
# Compare revenue across price bands (paid games only)
# =================================================================
paid = df[(df["Price_USD"] > 0)].dropna(subset=["Price_USD", "Revenue_M_USD"])
bands = pd.cut(paid["Price_USD"], bins=[0, 10, 20, 30, 45, 70])
band_stats = paid.groupby(bands, observed=True)["Revenue_M_USD"].agg(["count", "mean"]).round(1)

print("\n" + "=" * 64)
print("PRICE BANDS vs AVERAGE REVENUE (paid games)")
print("=" * 64)
print(band_stats.to_string())

best_band = band_stats["mean"].idxmax()
budget_band = band_stats.loc[[i for i in band_stats.index if i.right <= 10]]

if monetization == "Free-to-Play" or monetization == "Freemium":
    price_advice = "Free base game (revenue from in-game purchases)"
else:
    price_advice = (f"Within {best_band} USD - this band averages "
                    f"{band_stats.loc[best_band,'mean']:.0f}M revenue")

print(f"\n>>> PRICE ADVICE: {price_advice}")
print("    Note: the under-$10 band also performs well "
      f"({budget_band['mean'].iloc[0]:.0f}M avg) and the data's most common")
print("    market gap is 'affordable premium games' - a low-priced premium")
print("    title targets that gap directly.")

# =================================================================
# 4. RELEASE TIMING
# Use Social_Buzz_Score by research month as a seasonality proxy.
# CAVEAT: Research_Date is when the market data was gathered, not a
# release date - so this indicates when player attention runs hottest.
# =================================================================
monthly = (df.dropna(subset=["Month"])
             .groupby("Month")["Social_Buzz_Score"]
             .agg(["count", "mean"]).round(1))
monthly.index = [pd.Timestamp(2026, int(m), 1).strftime("%b") for m in monthly.index]

print("\n" + "=" * 64)
print("SOCIAL BUZZ BY MONTH (attention seasonality proxy)")
print("=" * 64)
print(monthly.to_string())

top_months = monthly["mean"].nlargest(3)
print(f"\n>>> RELEASE TIMING: aim for {', '.join(top_months.index)}")
print("    (highest average buzz; note Nov has only "
      f"{int(monthly.loc['Nov','count'])} data points, so weigh it lightly)")

# =================================================================
# 5. WHAT PLAYERS SAY IS MISSING (feature direction)
# =================================================================
gaps = df["Gap_Opportunity"].replace("", np.nan).dropna().value_counts().head(5)

print("\n" + "=" * 64)
print("TOP UNMET DEMANDS IN THE MARKET (design these in!)")
print("=" * 64)
for gap, n in gaps.items():
    print(f"  {n:>2} mentions - {gap}")

# =================================================================
# FINAL SUMMARY
# =================================================================
print("\n" + "#" * 64)
print("FINAL RECOMMENDATION")
print("#" * 64)
print(f"Build a {best_genre} game for {platform}, aimed at {audience},")
print(f"using a {monetization} model. {price_advice}.")
print(f"Target a release window of {', '.join(top_months.index[:2])},")
print("and differentiate with the top unmet demands listed above.")
