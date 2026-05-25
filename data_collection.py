import pandas as pd
import numpy as np
import os

# 106 countries with region assignments
COUNTRIES = [
    ("United States", "North America"), ("Canada", "North America"), ("Mexico", "North America"),
    ("United Kingdom", "Europe"), ("Germany", "Europe"), ("France", "Europe"), ("Italy", "Europe"),
    ("Spain", "Europe"), ("Netherlands", "Europe"), ("Sweden", "Europe"), ("Switzerland", "Europe"),
    ("Finland", "Europe"), ("Denmark", "Europe"), ("Norway", "Europe"), ("Belgium", "Europe"),
    ("Austria", "Europe"), ("Ireland", "Europe"), ("Poland", "Europe"), ("Czech Republic", "Europe"),
    ("Hungary", "Europe"), ("Romania", "Europe"), ("Greece", "Europe"), ("Portugal", "Europe"),
    ("Slovakia", "Europe"), ("Slovenia", "Europe"), ("Croatia", "Europe"), ("Bulgaria", "Europe"),
    ("Serbia", "Europe"), ("Ukraine", "Europe"), ("Russia", "Europe"), ("Iceland", "Europe"),
    ("Luxembourg", "Europe"), ("Estonia", "Europe"), ("Latvia", "Europe"), ("Lithuania", "Europe"),
    ("Malta", "Europe"), ("Cyprus", "Europe"),
    ("China", "Asia-Pacific"), ("Japan", "Asia-Pacific"), ("South Korea", "Asia-Pacific"),
    ("Singapore", "Asia-Pacific"), ("Australia", "Asia-Pacific"), ("New Zealand", "Asia-Pacific"),
    ("India", "Asia-Pacific"), ("Indonesia", "Asia-Pacific"), ("Malaysia", "Asia-Pacific"),
    ("Thailand", "Asia-Pacific"), ("Vietnam", "Asia-Pacific"), ("Philippines", "Asia-Pacific"),
    ("Pakistan", "Asia-Pacific"), ("Bangladesh", "Asia-Pacific"), ("Sri Lanka", "Asia-Pacific"),
    ("Taiwan", "Asia-Pacific"), ("Hong Kong", "Asia-Pacific"), ("Mongolia", "Asia-Pacific"),
    ("Kazakhstan", "Asia-Pacific"), ("Uzbekistan", "Asia-Pacific"), ("Myanmar", "Asia-Pacific"),
    ("Cambodia", "Asia-Pacific"), ("Laos", "Asia-Pacific"), ("Nepal", "Asia-Pacific"),
    ("Brazil", "Latin America"), ("Argentina", "Latin America"), ("Chile", "Latin America"),
    ("Colombia", "Latin America"), ("Peru", "Latin America"), ("Ecuador", "Latin America"),
    ("Uruguay", "Latin America"), ("Paraguay", "Latin America"), ("Bolivia", "Latin America"),
    ("Venezuela", "Latin America"), ("Costa Rica", "Latin America"), ("Panama", "Latin America"),
    ("Guatemala", "Latin America"), ("Dominican Republic", "Latin America"), ("Jamaica", "Latin America"),
    ("Saudi Arabia", "Middle East & Africa"), ("United Arab Emirates", "Middle East & Africa"),
    ("Qatar", "Middle East & Africa"), ("Kuwait", "Middle East & Africa"), ("Israel", "Middle East & Africa"),
    ("Turkey", "Middle East & Africa"), ("Iran", "Middle East & Africa"), ("Iraq", "Middle East & Africa"),
    ("Jordan", "Middle East & Africa"), ("Lebanon", "Middle East & Africa"), ("Oman", "Middle East & Africa"),
    ("Bahrain", "Middle East & Africa"), ("Egypt", "Middle East & Africa"), ("South Africa", "Middle East & Africa"),
    ("Nigeria", "Middle East & Africa"), ("Kenya", "Middle East & Africa"), ("Morocco", "Middle East & Africa"),
    ("Ghana", "Middle East & Africa"), ("Ethiopia", "Middle East & Africa"), ("Tanzania", "Middle East & Africa"),
    ("Rwanda", "Middle East & Africa"), ("Senegal", "Middle East & Africa"), ("Algeria", "Middle East & Africa"),
    ("Tunisia", "Middle East & Africa"), ("Angola", "Middle East & Africa"), ("Mozambique", "Middle East & Africa"),
    ("Zimbabwe", "Middle East & Africa"), ("Uganda", "Middle East & Africa"),
    ("Botswana", "Middle East & Africa"), ("Namibia", "Middle East & Africa"),
]

GDP_ANCHORS = {
    "United States": 70000, "Singapore": 65000, "Switzerland": 92000, "Norway": 75000,
    "Ireland": 100000, "Luxembourg": 120000, "Germany": 51000, "United Kingdom": 46000,
    "Japan": 40000, "Australia": 55000, "Canada": 52000, "China": 12000, "India": 2500,
    "Brazil": 7500, "Kenya": 2000, "Nigeria": 2200, "Ethiopia": 1100, "Bangladesh": 2600,
}

PILLAR_NAMES = [
    "talent_pillar",
    "infrastructure_pillar",
    "research_pillar",
    "policy_pillar",
    "adoption_pillar",
]

TARGET_GDP_CORR = 0.812


def create_sample_data():
    assert len(COUNTRIES) == 106, f"Expected 106 countries, got {len(COUNTRIES)}"

    np.random.seed(42)
    countries = [c[0] for c in COUNTRIES]
    regions = [c[1] for c in COUNTRIES]

    gdp_per_capita = []
    for c in countries:
        if c in GDP_ANCHORS:
            gdp_per_capita.append(GDP_ANCHORS[c])
        else:
            gdp_per_capita.append(int(np.random.randint(2500, 55000)))

    gdp_arr = np.array(gdp_per_capita, dtype=float)
    gdp_z = (gdp_arr - gdp_arr.mean()) / gdp_arr.std()

    noise = np.random.normal(0, 1, len(countries))
    noise_z = (noise - noise.mean()) / noise.std()
    readiness_z = TARGET_GDP_CORR * gdp_z + np.sqrt(1 - TARGET_GDP_CORR**2) * noise_z

    readiness = 35 + readiness_z * 18
    ai_readiness = np.clip(readiness, 20, 98).round(1)

    internet = np.clip(25 + ai_readiness * 0.65 + np.random.normal(0, 6, len(countries)), 15, 99).round(1)
    gov_effectiveness = np.round(-2 + (ai_readiness / 100) * 5 + np.random.normal(0, 0.3, len(countries)), 2)

    pillar_data = {}
    for pillar in PILLAR_NAMES:
        offset = np.random.uniform(-8, 8, len(countries))
        scores = np.clip(ai_readiness + offset + np.random.normal(0, 3, len(countries)), 15, 98).round(1)
        pillar_data[pillar] = scores

    df = pd.DataFrame({
        "country": countries,
        "region": regions,
        "ai_readiness_score": ai_readiness,
        "gdp_per_capita_usd": gdp_per_capita,
        "internet_penetration_pct": internet,
        "government_effectiveness": gov_effectiveness,
        **pillar_data,
        "year": 2024,
    })

    all_years = [df]
    for year in [2021, 2022, 2023]:
        df_year = df.copy()
        df_year["ai_readiness_score"] = (df_year["ai_readiness_score"] - (2024 - year) * 1.5).clip(15, 95)
        for pillar in PILLAR_NAMES:
            df_year[pillar] = (df_year[pillar] - (2024 - year) * 1.2).clip(15, 95)
        df_year["year"] = year
        all_years.append(df_year)

    return pd.concat(all_years, ignore_index=True)


def main():
    os.makedirs("data", exist_ok=True)
    df = create_sample_data()
    df.to_csv("data/ai_readiness_data.csv", index=False)
    corr_2024 = df[df["year"] == 2024]["ai_readiness_score"].corr(df[df["year"] == 2024]["gdp_per_capita_usd"])
    print(f"Saved {len(df)} records")
    print(f"Countries: {df['country'].nunique()}")
    print(f"GDP correlation (2024): {corr_2024:.3f}")
    print(f"Top 5: {df[df['year'] == 2024].nlargest(5, 'ai_readiness_score')['country'].tolist()}")


if __name__ == "__main__":
    main()
