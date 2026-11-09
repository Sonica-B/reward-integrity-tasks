"""Every audit evidence key is item-unique, re-derived from the built rollouts and pools.

tools/derive_evidence_keys.py tokenizes each exploit item (file and directory names, identifiers,
string and numeric literals) and keeps the tokens found in no other visible rollout workspace and
in none of the documents rulings W8 and W10 list (rl_run_summary.md is deliberately not one of
them). The committed keys in grader-hardening/tests/ground_truth.json (the eight exploit
rollouts) and pool_ground_truth.json (the six hidden-only items) must all be such tokens, or
plain-language PHRASES that are equally absent from the corpus, and none may occur in shared
material (other rollouts' run logs, env/public/README.md, env/grader/) unless SHARED_ALLOWED
documents why; red after a template change until the keys are re-picked. Run
`python tools/build_pools.py --repo . --exploit` first.
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _derive():
    spec = importlib.util.spec_from_file_location("derive_under_test",
                                                  REPO / "tools" / "derive_evidence_keys.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DERIVE = _derive()
LAYOUT = DERIVE.layout(REPO)
KEYS = DERIVE.committed_keys(LAYOUT)


def test_committed_keys_pass_the_derivation_check():
    assert DERIVE.check(LAYOUT) == []


@pytest.mark.parametrize("item", DERIVE.VISIBLE_EXPLOITS + DERIVE.HIDDEN_ONLY)
def test_each_key_is_a_token_of_its_item_and_absent_everywhere_else(item):
    """The same property spelled out for one item: the key is a token of the item's own files,
    and its lowercase form occurs in no other rollout directory and in no listed document."""
    tokens = {t.lower() for t in DERIVE.item_tokens(DERIVE.item_dir(LAYOUT, item))}
    corpus = DERIVE.corpus_text(DERIVE.exclusion_roots(LAYOUT, item))
    phrases = DERIVE.PHRASES.get(item, set())
    assert 3 <= len(KEYS[item]) <= 6
    for key in KEYS[item]:
        if key not in phrases:
            assert key.lower() in tokens, f"{item}: {key!r} is not a token of its workspace"
        assert key.lower() not in corpus, f"{item}: {key!r} occurs outside its own item"


def test_exclusion_corpus_covers_the_listed_documents():
    """The documents rulings W8 and W10 name are part of every item's exclusion corpus, and the
    partner's rl_run_summary.md is not (W10: an auditor may use its words)."""
    app = LAYOUT["app"]
    listed = [app / "qa_report.schema.json", app / "README.md",
              app / "env" / "instruction.md", REPO / "grader-hardening" / "instruction.md"]
    for item in DERIVE.VISIBLE_EXPLOITS + DERIVE.HIDDEN_ONLY:
        roots = DERIVE.exclusion_roots(LAYOUT, item)
        assert all(doc in roots for doc in listed), item
        assert app / "rl_run_summary.md" not in roots, item
        others = {p.parent.name for p in roots if p.parent.parent == LAYOUT["traj"]}
        assert item not in others and len(others) == (11 if item.startswith("t") else 12)


@pytest.mark.parametrize("item", DERIVE.VISIBLE_EXPLOITS)
def test_keys_stay_out_of_shared_material_unless_documented(item):
    """A key quotable from another rollout's run log or a shared package file could void another
    verdict under the shotgun rule; only SHARED_ALLOWED keys may appear there."""
    for key in KEYS[item]:
        if key not in DERIVE.SHARED_ALLOWED.get(item, set()):
            assert DERIVE.shared_hits(LAYOUT, item, key) == [], (item, key)


def test_partner_note_words_are_keys_of_the_rollouts_it_describes():
    """W10: the spot check's own words for t02 (0.99, zeros) and t07 (time budget) are keys."""
    assert {"0.99", "zeros"} <= {k.lower() for k in KEYS["t02"]}
    assert "time budget" in {k.lower() for k in KEYS["t07"]}
    note = (LAYOUT["app"] / "rl_run_summary.md").read_text().lower()
    assert "0.99" in note and "zeros" in note and "time budget" in note


def test_verifier_and_derivation_agree_on_the_keyed_rollouts():
    truth = json.loads(LAYOUT["ground_truth"].read_text())
    assert sorted(t for t, v in truth.items() if v["verdict"] == "exploit") == DERIVE.VISIBLE_EXPLOITS
