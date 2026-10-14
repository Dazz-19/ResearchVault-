import json
from collections import Counter


ALLOWED_CATEGORIES = {"answerable", "unanswerable", "multi_paper", "trick", "ambiguous"}


KNOWN_PAPERS = {
    "NIPS-2017-attention-is-all-you-need-Paper",
    "chain_of_though_prmpt",
    "training_lang_models",
}

REQUIRED_FIELDS = {
    "id", "question", "category", "should_abstain",
    "relevant_pages", "reference_answer", "keywords",
}


def check_case(case):
    """Raise ValueError (naming the question id) if one test case is invalid."""
    cid = case.get("id", "?")

    missing = REQUIRED_FIELDS - case.keys()
    if missing:
        raise ValueError(f"Q{cid}: missing fields {sorted(missing)}")

    if case["category"] not in ALLOWED_CATEGORIES:
        raise ValueError(f"Q{cid}: unknown category {case['category']!r}")

    for rp in case["relevant_pages"]:
        if rp["paper"] not in KNOWN_PAPERS:
            raise ValueError(f"Q{cid}: unknown paper {rp['paper']!r}")
        if not isinstance(rp["page"], int) or rp["page"] < 1:
            raise ValueError(f"Q{cid}: page must be an int >= 1, got {rp['page']!r}")

    if case["category"] == "unanswerable":
        if case["relevant_pages"] or not case["should_abstain"]:
            raise ValueError(
                f"Q{cid}: unanswerable needs no relevant pages and should_abstain=true"
            )


def load_test_cases(path="data/eval/test_cases.jsonl"):
    """Read a JSONL file and return a list of validated test-case dicts."""
    cases = []
    seen_ids = set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_no}: invalid JSON ({e})") from e
            check_case(case)
            if case["id"] in seen_ids:
                raise ValueError(f"Q{case['id']}: duplicate id")
            seen_ids.add(case["id"])
            cases.append(case)
    return cases


if __name__ == "__main__":
    cases = load_test_cases()
    print(f"Loaded {len(cases)} questions")
    for category, n in Counter(c["category"] for c in cases).items():
        print(f"  {category}: {n}")