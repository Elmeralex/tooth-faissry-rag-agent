"""Maximum local anaesthetic dose calculator.

Dose arithmetic is exactly what an LLM should not do in its head, so the agent
calls this deterministic tool instead.

Reference values are the commonly cited dental maximum recommended doses
(e.g. Malamed, Handbook of Local Anesthesia). Sources differ, so verify them
against your own clinical references before relying on the numbers.
Educational use only.
"""

from __future__ import annotations

import math
from typing import Any

from agent.tools.base import Tool

ANAESTHETICS: dict[str, dict[str, float]] = {
    "lidocaine": {"mg_per_kg": 4.4, "max_mg": 300, "default_percent": 2.0},
    "mepivacaine": {"mg_per_kg": 4.4, "max_mg": 300, "default_percent": 3.0},
    "articaine": {"mg_per_kg": 7.0, "max_mg": 500, "default_percent": 4.0},
    "prilocaine": {"mg_per_kg": 8.0, "max_mg": 600, "default_percent": 4.0},
    "bupivacaine": {"mg_per_kg": 1.3, "max_mg": 90, "default_percent": 0.5},
}
ALIASES = {
    "lignocaine": "lidocaine",
    "xylocaine": "lidocaine",
    "carbocaine": "mepivacaine",
    "septocaine": "articaine",
    "citanest": "prilocaine",
    "marcaine": "bupivacaine",
}
DEFAULT_CARTRIDGE_ML = 1.8


def calculate_max_dose(
    anaesthetic: str,
    weight_kg: float,
    concentration_percent: float | None = None,
    cartridge_ml: float = DEFAULT_CARTRIDGE_ML,
) -> dict[str, Any]:
    key = anaesthetic.strip().lower()
    key = ALIASES.get(key, key)
    if key not in ANAESTHETICS:
        raise ValueError(f"Unknown anaesthetic '{anaesthetic}'. Supported: {', '.join(ANAESTHETICS)}")
    if not 1 <= weight_kg <= 250:
        raise ValueError("weight_kg must be between 1 and 250")
    if not 0 < cartridge_ml <= 5:
        raise ValueError("cartridge_ml must be between 0 and 5")

    ref = ANAESTHETICS[key]
    percent = concentration_percent or ref["default_percent"]
    if not 0 < percent <= 10:
        raise ValueError("concentration_percent must be between 0 and 10")

    weight_based_mg = weight_kg * ref["mg_per_kg"]
    max_mg = min(weight_based_mg, ref["max_mg"])
    mg_per_cartridge = percent * 10 * cartridge_ml  # 1% solution = 10 mg/mL
    # Round down, never up: 7.39 cartridges means 7.3, not 7.4.
    max_cartridges = math.floor(max_mg / mg_per_cartridge * 10) / 10

    return {
        "anaesthetic": key,
        "weight_kg": weight_kg,
        "concentration_percent": percent,
        "cartridge_ml": cartridge_ml,
        "mg_per_kg": ref["mg_per_kg"],
        "absolute_max_mg": ref["max_mg"],
        "weight_based_max_mg": round(weight_based_mg, 1),
        "max_dose_mg": round(max_mg, 1),
        "limited_by": "absolute maximum" if ref["max_mg"] < weight_based_mg else "body weight",
        "mg_per_cartridge": round(mg_per_cartridge, 1),
        "max_cartridges": max_cartridges,
        "note": "Educational estimate from reference maximums; clinical judgement and patient factors may lower it.",
    }


TOOL = Tool(
    name="calculate_max_dose",
    description=(
        "Calculate the maximum recommended dose of a dental local anaesthetic for a patient's body weight, "
        "in mg and in cartridges. Always use this instead of doing dose arithmetic yourself."
    ),
    parameters={
        "type": "object",
        "properties": {
            "anaesthetic": {"type": "string", "enum": list(ANAESTHETICS)},
            "weight_kg": {"type": "number", "description": "Patient body weight in kilograms"},
            "concentration_percent": {
                "type": "number",
                "description": "Solution strength, e.g. 2 for lidocaine 2%. Omit to use the usual dental strength.",
            },
            "cartridge_ml": {"type": "number", "description": "Cartridge volume in mL (default 1.8)"},
        },
        "required": ["anaesthetic", "weight_kg"],
    },
    func=calculate_max_dose,
)
