"""Sector keyword rules.

The sector rules are deliberately narrow: they only need to be precise enough to produce
training labels. The classifier in `radar.classify` learns to label everything else.
"""
from __future__ import annotations

import re

SECTOR_RULES: dict[str, list[str]] = {
    "Construction & Civil Works": [
        r"construction of", r"civil works?", r"road(s)? (construction|rehabilitation|upgrad)",
        r"paving", r"bridge", r"storm ?water", r"bulk earthworks?", r"resurfacing",
        r"gravel road", r"new (school|clinic|hall|building)", r"\bcidb\b",
    ],
    "Building Maintenance & Repairs": [
        r"building maintenance", r"maintenance and repair", r"repairs? (to|of) (the )?(building|roof|offices|facilit)",
        r"renovation", r"refurbish", r"painting", r"waterproofing", r"plumbing",
        r"\bhvac\b", r"air[- ]?condition", r"\blifts?\b|elevator",
    ],
    "Cleaning, Hygiene & Waste": [
        r"cleaning services?", r"hygiene", r"sanitary", r"pest control", r"fumigation",
        r"waste (removal|management|collection)", r"refuse", r"landfill", r"grass cutting",
        r"garden(ing)? services", r"horticultur", r"landscap",
    ],
    "Security Services": [
        r"security services?", r"guarding", r"armed response", r"\bcctv\b",
        r"access control", r"security guards?", r"alarm",
    ],
    "ICT & Software": [
        r"\bict\b", r"software", r"licen[cs]es?", r"laptops?", r"computers?", r"tablets?",
        r"network(ing)?", r"servers?", r"\bwi-?fi\b", r"data cent(er|re)", r"cyber",
        r"microsoft", r"information technology", r"\berp\b", r"website", r"printers?",
    ],
    "Professional & Consulting": [
        r"consultan", r"advisory", r"audit", r"actuarial", r"legal services", r"attorneys",
        r"feasibility", r"research", r"valuation", r"forensic", r"panel of (consultants|experts)",
        r"engineering services", r"professional services", r"town planning",
        r"verification", r"law firms?", r"environmental (impact|management|authori[sz]ation)",
        r"actuar", r"strategy", r"business plan", r"evaluation study",
    ],
    "Health & Medical": [
        r"medical", r"pharmac", r"clinical", r"hospital", r"laboratory", r"surgical",
        r"\bppe\b", r"medicines?", r"health practitioners?", r"dental", r"nursing",
        r"employee (assistance|wellness)", r"occupational health",
    ],
    "Transport, Fleet & Logistics": [
        r"vehicles?", r"\bfleet\b", r"truck", r"bus(es)?\b", r"transport(ation)? services",
        r"courier", r"tyres?", r"logistics", r"shuttle", r"car hire|vehicle rental",
    ],
    "Catering, Events & Travel": [
        r"catering", r"accommodation", r"venue", r"conference", r"\bevents?\b",
        r"travel (management|agency)", r"food (supply|parcels)", r"meals",
    ],
    "Electrical & Energy": [
        r"electrical", r"electrification", r"solar", r"\bpv\b", r"generators?",
        r"transformers?", r"substation", r"street ?lights?|high ?mast", r"energy",
        r"cables?", r"switchgear", r"\bups\b",
    ],
    "Water & Sanitation": [
        r"water (supply|reticulation|treatment|meters?|tankers?)", r"sewer", r"sewage",
        r"sanitation", r"boreholes?", r"reservoir", r"pipeline", r"pump station",
        r"wastewater", r"\bvip\b toilets?",
    ],
    "Supplies, Printing & Uniforms": [
        r"stationery", r"printing", r"uniforms?", r"protective clothing", r"furniture",
        r"office supplies", r"consumables", r"toner", r"branding", r"promotional",
    ],
    "Property & Leasing": [
        r"lease (of )?(the )?(premises|property|office|land|space|building)", r"leasing of",
        r"office (space|accommodation)", r"rental of (office|premises|property|land)",
        r"property (management|valuation)", r"to lease", r"letting",
    ],
    "Machinery, Plant & Spares": [
        r"spares?", r"spare parts", r"tractors?", r"plant hire", r"yellow (plant|fleet)",
        r"excavators?", r"tlb", r"graders?", r"cranes?", r"forklifts?", r"pumps?",
        r"valves?", r"bearings?", r"tools?", r"machinery", r"conveyor", r"compressors?",
    ],
    "Training & Development": [
        r"training", r"learnership", r"skills development", r"capacity building",
        r"accredited", r"bursar", r"mentorship", r"coaching",
    ],
}

SECTORS = list(SECTOR_RULES)
OTHER = "Other"

_COMPILED = {s: re.compile("|".join(f"(?:{p})" for p in pats), re.I)
             for s, pats in SECTOR_RULES.items()}


def rule_hits(text: str) -> dict[str, int]:
    """Number of distinct keyword hits per sector."""
    hits = {}
    for sector, rx in _COMPILED.items():
        n = len({m.group(0).lower() for m in rx.finditer(text)})
        if n:
            hits[sector] = n
    return hits


def rule_label(text: str) -> str | None:
    """A sector label when the rules are unambiguous, else None."""
    hits = rule_hits(text)
    if not hits:
        return None
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def tied_sectors(text: str) -> list[str]:
    """Sectors sharing the top rule score, when more than one does."""
    hits = rule_hits(text)
    if len(hits) < 2:
        return []
    top = max(hits.values())
    tied = [s for s, n in hits.items() if n == top]
    return tied if len(tied) > 1 else []


def mask_rules(text: str) -> str:
    """Remove the rule keywords, so a model trained on rule labels has to learn context."""
    for rx in _COMPILED.values():
        text = rx.sub(" ", text)
    return text
