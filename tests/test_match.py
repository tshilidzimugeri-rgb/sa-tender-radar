import pandas as pd

from radar.match import Profile, score


def tenders():
    return pd.DataFrame([
        {"id": 1, "description": "Office cleaning services", "category": "", "buyer": "A",
         "sector": "Cleaning, Hygiene & Waste", "sector_2": "Security Services",
         "province": "Gauteng", "closes": pd.Timestamp("2026-11-01")},
        {"id": 2, "description": "Office cleaning services", "category": "", "buyer": "B",
         "sector": "Cleaning, Hygiene & Waste", "sector_2": "Security Services",
         "province": "Limpopo", "closes": pd.Timestamp("2026-11-01")},
        {"id": 3, "description": "Supply of laptops", "category": "", "buyer": "C",
         "sector": "ICT & Software", "sector_2": "Supplies, Printing & Uniforms",
         "province": "Gauteng", "closes": pd.Timestamp("2026-11-01")},
        {"id": 4, "description": "Cleaning of reservoirs", "category": "", "buyer": "D",
         "sector": "Water & Sanitation", "sector_2": "Cleaning, Hygiene & Waste",
         "province": "National", "closes": pd.Timestamp("2026-11-01")},
    ])


def test_ranking_prefers_sector_and_province():
    p = Profile(sectors=["Cleaning, Hygiene & Waste"], provinces=["Gauteng"],
                keywords=["office cleaning"])
    ranked = score(tenders(), p)
    assert list(ranked["id"])[:2] == [1, 2]
    assert ranked.set_index("id").loc[1, "score"] == 100
    assert ranked.set_index("id").loc[3, "score"] < 50


def test_second_choice_sector_gets_partial_credit():
    p = Profile(sectors=["Cleaning, Hygiene & Waste"])
    s = score(tenders(), p).set_index("id")["score"]
    assert s[1] == 100 and s[4] == 50 and s[3] == 0


def test_exclude_zeroes_score():
    p = Profile(sectors=["Cleaning, Hygiene & Waste"], exclude=["reservoir"])
    s = score(tenders(), p).set_index("id")["score"]
    assert s[4] == 0
