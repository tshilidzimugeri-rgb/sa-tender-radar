from radar.taxonomy import mask_rules, rule_label, tied_sectors


def test_clear_descriptions_get_a_rule_label():
    assert rule_label("Provision of security guarding services for 36 months") == "Security Services"
    assert rule_label("Supply and delivery of 60 laptops") == "ICT & Software"
    assert rule_label("Office cleaning services at the regional office") == "Cleaning, Hygiene & Waste"


def test_no_keywords_or_ties_give_no_label():
    assert rule_label("Appointment of a service provider for the period of 36 months") is None
    text = "Supply of laptops and security guards"
    assert rule_label(text) is None
    assert set(tied_sectors(text)) == {"ICT & Software", "Security Services"}


def test_mask_removes_rule_keywords():
    masked = mask_rules("provision of security guarding services at the head office")
    assert "security" not in masked and "head office" in masked
