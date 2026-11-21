import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "golden_dataset" / "ragas_golden_dataset.json"
DOCS_DIR = ROOT / "data" / "documents"

def load_dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))

def test_all_five_source_documents_exist():
    expected = {
        "01_policy_overview.md",
        "02_benefits_and_limits.md",
        "03_waiting_periods_and_exclusions.md",
        "04_claims_procedure.md",
        "05_premium_and_servicing.md",
    }
    assert expected.issubset({p.name for p in DOCS_DIR.glob("*.md")})

def test_dataset_has_exactly_15_records():
    data = load_dataset()
    assert data["count"] == 15
    assert len(data["records"]) == 15

def test_question_ids_are_unique():
    records = load_dataset()["records"]
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))

def test_category_distribution_is_correct():
    data = load_dataset()
    observed = {}
    for record in data["records"]:
        observed[record["category"]] = observed.get(record["category"], 0) + 1
    assert observed == data["categories"]

def test_required_fields_are_present_and_nonempty():
    required = {"id", "category", "user_input", "reference",
                "reference_contexts", "source_document_ids", "expected_behavior"}
    for record in load_dataset()["records"]:
        assert required.issubset(record.keys())
        assert record["user_input"].strip()
        assert record["reference"].strip()
        assert record["expected_behavior"].strip()
        assert isinstance(record["reference_contexts"], list)
        assert isinstance(record["source_document_ids"], list)

def test_unanswerable_questions_have_no_reference_contexts():
    records = load_dataset()["records"]
    unanswerable = [r for r in records if r["category"] == "unanswerable"]
    assert len(unanswerable) == 2
    assert all(r["reference_contexts"] == [] for r in unanswerable)

def test_all_cited_source_documents_exist():
    available = {p.name for p in DOCS_DIR.glob("*.md")}
    for record in load_dataset()["records"]:
        assert set(record["source_document_ids"]).issubset(available)

def test_every_answerable_record_has_reference_context():
    records = load_dataset()["records"]
    answerable = [r for r in records if r["category"] != "unanswerable"]
    assert all(len(r["reference_contexts"]) > 0 for r in answerable)
