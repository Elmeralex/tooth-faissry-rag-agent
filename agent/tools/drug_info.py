"""Drug label lookup through the public openFDA API (no API key needed)."""

from __future__ import annotations

from typing import Any

import httpx

from agent.tools.base import Tool

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
SECTIONS = (
    "boxed_warning",
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings",
    "drug_interactions",
    "adverse_reactions",
)
MAX_SECTION_CHARS = 500
CANDIDATE_LABELS = 10  # fetch several labels and pick the closest name match


def lookup_drug_info(drug_name: str, *, client: httpx.Client | None = None) -> dict[str, Any]:
    name = drug_name.replace('"', "").strip()
    if not name:
        raise ValueError("drug_name is empty")

    params = {
        "search": f'openfda.generic_name:"{name}" OR openfda.brand_name:"{name}"',
        "limit": CANDIDATE_LABELS,
    }
    http = client or httpx.Client(timeout=10)
    try:
        response = http.get(OPENFDA_LABEL_URL, params=params)
    finally:
        if client is None:
            http.close()

    if response.status_code == 404:  # openFDA answers 404 when nothing matches
        return {"drug": name, "found": False, "message": "No FDA drug label found for this name."}
    response.raise_for_status()

    label = _best_match(name, response.json()["results"])
    openfda = label.get("openfda", {})
    sections = {
        section: _shorten(" ".join(label[section]))
        for section in SECTIONS
        if label.get(section)
    }
    return {
        "drug": name,
        "found": True,
        "generic_name": _first(openfda.get("generic_name")),
        "brand_name": _first(openfda.get("brand_name")),
        "sections": sections,
        "source": "openFDA drug label (US FDA)",
    }


def _best_match(name: str, labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer an exact generic/brand name match ("amoxicillin" over "amoxicillin and clavulanate")."""
    wanted = name.lower()
    for label in labels:
        openfda = label.get("openfda", {})
        names = openfda.get("generic_name", []) + openfda.get("brand_name", [])
        if wanted in (n.lower() for n in names):
            return label
    return labels[0]


def _first(values: list[str] | None) -> str | None:
    return values[0] if values else None


def _shorten(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_SECTION_CHARS:
        return text
    return text[:MAX_SECTION_CHARS].rsplit(" ", 1)[0] + " ..."


TOOL = Tool(
    name="lookup_drug_info",
    description=(
        "Look up the official US FDA label for a drug: indications, dosage, contraindications, warnings, "
        "drug interactions and adverse reactions. Use for antibiotics, analgesics, anaesthetics, etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "drug_name": {"type": "string", "description": "Generic or brand name, e.g. 'amoxicillin'"},
        },
        "required": ["drug_name"],
    },
    func=lookup_drug_info,
)
