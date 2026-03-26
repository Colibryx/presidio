"""Tests for batch /anonymize when text and analyzer_results are lists."""

import pytest

from app import create_app


@pytest.fixture(name="client")
def fixture_client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_when_text_is_list_then_returns_list_of_anonymized_payloads(client):
    payload = {
        "text": ["My name is Alice", "Reach me at 555-123-4567"],
        "analyzer_results": [
            [
                {
                    "entity_type": "PERSON",
                    "start": 11,
                    "end": 16,
                    "score": 0.9,
                }
            ],
            [
                {
                    "entity_type": "PHONE_NUMBER",
                    "start": 12,
                    "end": 24,
                    "score": 0.85,
                }
            ],
        ],
    }
    resp = client.post("/anonymize", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert "text" in data[0] and "text" in data[1]
    assert "Alice" not in data[0]["text"]
    assert "555" not in data[1]["text"]


def test_when_batch_lengths_mismatch_then_400(client):
    payload = {
        "text": ["a", "b"],
        "analyzer_results": [[]],
    }
    resp = client.post("/anonymize", json=payload)
    assert resp.status_code == 400
    assert "same length" in resp.get_json()["error"].lower()


def test_when_batch_text_but_flat_analyzer_results_then_400(client):
    payload = {
        "text": ["a", "b"],
        "analyzer_results": [
            {"entity_type": "PERSON", "start": 0, "end": 1, "score": 0.5}
        ],
    }
    resp = client.post("/anonymize", json=payload)
    assert resp.status_code == 400


def test_when_batch_analyzer_entry_not_list_then_400(client):
    payload = {
        "text": ["a", "b"],
        "analyzer_results": [
            [],
            {"entity_type": "PERSON", "start": 0, "end": 1, "score": 0.5},
        ],
    }
    resp = client.post("/anonymize", json=payload)
    assert resp.status_code == 400
