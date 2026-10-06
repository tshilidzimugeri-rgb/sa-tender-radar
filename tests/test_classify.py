import pandas as pd

from radar.classify import apply, train

EXAMPLES = {
    "Security Services": ["security guarding services at {}", "armed response for {}"],
    "ICT & Software": ["supply of laptops for {}", "software licences for {}"],
    "Cleaning, Hygiene & Waste": ["cleaning services at {}", "waste removal at {}"],
}
PLACES = ["the head office", "regional offices", "district clinics", "the depot",
          "provincial offices", "the warehouse", "satellite offices", "the airport"]


def corpus():
    rows = []
    buyers = {"Security Services": "police service", "ICT & Software": "it department",
              "Cleaning, Hygiene & Waste": "public works"}
    for sector, templates in EXAMPLES.items():
        for t in templates:
            for p in PLACES * 3:
                rows.append({"description": t.format(p), "category": f"{sector} services",
                             "buyer": buyers[sector]})
    return pd.DataFrame(rows)


def test_model_labels_tenders_without_keywords():
    model, report = train(corpus(), evaluate=False)
    assert report["training_examples"] > 100
    new = pd.DataFrame([{"description": "appointment of a service provider at the depot",
                         "category": "ICT & Software services", "buyer": "it department"}])
    out = apply(model, new)
    assert out["sector"].iloc[0] == "ICT & Software"
    assert out["sector_source"].iloc[0] == "model"


def test_rule_labels_win_over_model():
    model, _ = train(corpus(), evaluate=False)
    new = pd.DataFrame([{"description": "cleaning services at the depot",
                         "category": "ICT & Software services", "buyer": "it department"}])
    out = apply(model, new)
    assert out["sector"].iloc[0] == "Cleaning, Hygiene & Waste"
    assert out["sector_confidence"].iloc[0] == 1.0
