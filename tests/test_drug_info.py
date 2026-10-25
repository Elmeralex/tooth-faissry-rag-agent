import httpx

from agent.tools.drug_info import MAX_SECTION_CHARS, lookup_drug_info

LABEL = {
    "results": [
        {
            "openfda": {"generic_name": ["AMOXICILLIN"], "brand_name": ["Amoxil"]},
            "contraindications": ["Serious hypersensitivity to amoxicillin or other beta-lactams."],
            "warnings": ["word " * 400],
            "drug_interactions": ["Probenecid", "may increase amoxicillin levels."],
        }
    ]
}


def client_returning(status, payload, seen):
    def handler(request):
        seen.append(request)
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_parses_label_sections():
    seen = []
    result = lookup_drug_info("amoxicillin", client=client_returning(200, LABEL, seen))

    assert 'openfda.generic_name:"amoxicillin"' in seen[0].url.params["search"]
    assert result["found"] is True
    assert result["generic_name"] == "AMOXICILLIN"
    assert result["brand_name"] == "Amoxil"
    assert result["sections"]["drug_interactions"] == "Probenecid may increase amoxicillin levels."
    assert len(result["sections"]["warnings"]) <= MAX_SECTION_CHARS + 4
    assert "boxed_warning" not in result["sections"]


def test_prefers_exact_name_over_combination_products():
    combo = {"openfda": {"generic_name": ["AMOXICILLIN AND CLAVULANATE POTASSIUM"]}, "warnings": ["combo"]}
    plain = {"openfda": {"generic_name": ["AMOXICILLIN"]}, "warnings": ["plain"]}
    payload = {"results": [combo, plain]}
    result = lookup_drug_info("amoxicillin", client=client_returning(200, payload, []))
    assert result["generic_name"] == "AMOXICILLIN"

    payload = {"results": [combo]}
    result = lookup_drug_info("amoxicillin", client=client_returning(200, payload, []))
    assert result["generic_name"] == "AMOXICILLIN AND CLAVULANATE POTASSIUM"  # falls back to the first label


def test_not_found_is_a_normal_result():
    result = lookup_drug_info("notadrug", client=client_returning(404, {"error": {}}, []))
    assert result == {"drug": "notadrug", "found": False, "message": "No FDA drug label found for this name."}


def test_quotes_cannot_break_the_query():
    seen = []
    lookup_drug_info('ibu"profen', client=client_returning(200, LABEL, seen))
    assert 'generic_name:"ibuprofen"' in seen[0].url.params["search"]
