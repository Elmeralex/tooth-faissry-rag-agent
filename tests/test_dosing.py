import pytest

from agent.tools.dosing import calculate_max_dose


def test_weight_limited_lidocaine():
    result = calculate_max_dose("lidocaine", 60)
    assert result["max_dose_mg"] == 264.0  # 60 kg x 4.4 mg/kg
    assert result["mg_per_cartridge"] == 36.0  # 2% x 10 mg/mL x 1.8 mL
    assert result["max_cartridges"] == 7.3  # 7.33 rounded down
    assert result["limited_by"] == "body weight"


def test_absolute_maximum_caps_heavy_patient():
    result = calculate_max_dose("lidocaine", 90)
    assert result["weight_based_max_mg"] == 396.0
    assert result["max_dose_mg"] == 300
    assert result["limited_by"] == "absolute maximum"


def test_cartridges_round_down_not_up():
    # articaine 4%, 1.7 mL cartridge = 68 mg; 70 kg x 7 = 490 mg -> 7.205... cartridges
    result = calculate_max_dose("articaine", 70, cartridge_ml=1.7)
    assert result["max_cartridges"] == 7.2


def test_alias_and_case_insensitive():
    assert calculate_max_dose("  Lignocaine ", 50)["anaesthetic"] == "lidocaine"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"anaesthetic": "cocaine", "weight_kg": 60},
        {"anaesthetic": "lidocaine", "weight_kg": 0},
        {"anaesthetic": "lidocaine", "weight_kg": 60, "cartridge_ml": -1},
        {"anaesthetic": "lidocaine", "weight_kg": 60, "concentration_percent": 50},
    ],
)
def test_invalid_input_raises(kwargs):
    with pytest.raises(ValueError):
        calculate_max_dose(**kwargs)
