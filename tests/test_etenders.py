import json

from radar.etenders import flatten, to_frame

RAW = {
    "id": 172765, "tender_No": "CCMA/2026/03/TSH", "type": "Request for Bid(Open-Tender)",
    "delivery": "28 Harrison Street - Johannesburg", "department": "CCMA",
    "date_Published": "2026-10-02T00:00:00", "closing_Date": "2026-10-26T11:00:00",
    "compulsory_briefing_session": "0001-01-01T00:00:00", "category": "Services: General",
    "description": "PROVISION OF  SECURITY\nSERVICES", "province": "Gauteng",
    "contactPerson": "Ms Example", "email": "x@example.org", "telephone": "011",
    "briefingVenue": None, "conditions": "N/A", "bf": "Yes", "bc": " NO",
    "eSubmission": False,
    "sd": [{"supportDocumentID": "abc", "fileName": "Bid doc.pdf", "extension": ".pdf",
            "active": True}],
}


def test_flatten_cleans_fields():
    row = flatten(RAW)
    assert row["description"] == "PROVISION OF SECURITY SERVICES"
    assert row["conditions"] == ""
    assert row["briefing"] is True and row["briefing_compulsory"] is False
    docs = json.loads(row["documents"])
    assert docs[0]["url"].endswith("blobName=abc.pdf&downloadedFileName=Bid%20doc.pdf")


def test_placeholder_dates_become_missing():
    df = to_frame([RAW])
    assert df["briefing_date"].isna().all()
    assert df["closes"].iloc[0].hour == 11
