"""
Unit tests for eval metric helpers.

No DB, no LLM, no network. Tests the pure-function metric calculators
imported from eval_rag.py and eval_retrieval.py indirectly through
importing the scripts as modules.
"""

import sys
import pathlib

# Make scripts importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scripts.eval_rag import token_f1, exact_match, citation_precision, abstention_correct


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------

def test_f1_exact():
    assert token_f1("the quick brown fox", "the quick brown fox") == 1.0


def test_f1_partial():
    score = token_f1("Alexander Bell", "Alexander Graham Bell")
    assert 0 < score < 1


def test_f1_no_overlap():
    assert token_f1("cat", "dog") == 0.0


def test_f1_empty_pred():
    assert token_f1("", "some answer") == 0.0


def test_f1_empty_gold():
    assert token_f1("some answer", "") == 0.0


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

def test_em_exact():
    assert exact_match("Alexander Bell", "Alexander Bell") is True


def test_em_case_insensitive():
    assert exact_match("ALEXANDER BELL", "alexander bell") is True


def test_em_whitespace_normalized():
    assert exact_match("Alexander  Bell", "Alexander Bell") is True


def test_em_different():
    assert exact_match("Bell", "Edison") is False


# ---------------------------------------------------------------------------
# citation_precision
# ---------------------------------------------------------------------------

def test_cite_p_all_valid():
    assert citation_precision("Answer [c1] and [c2].", {"c1", "c2"}) == 1.0


def test_cite_p_some_invalid():
    score = citation_precision("Answer [c1] and [fake].", {"c1"})
    assert score == 0.5


def test_cite_p_no_citations():
    assert citation_precision("No citations here.", {"c1"}) == 1.0


def test_cite_p_all_invalid():
    assert citation_precision("Answer [fake1] [fake2].", {"c1"}) == 0.0


# ---------------------------------------------------------------------------
# abstention_correct
# ---------------------------------------------------------------------------

def test_abstain_correct_when_should():
    assert abstention_correct(abstained=True, gold_answer="") is True


def test_abstain_wrong_when_should_not():
    assert abstention_correct(abstained=True, gold_answer="some answer") is False


def test_answer_correct_when_answerable():
    assert abstention_correct(abstained=False, gold_answer="some answer") is True


def test_answer_wrong_when_unanswerable():
    assert abstention_correct(abstained=False, gold_answer="") is False


# ---------------------------------------------------------------------------
# Questions file format
# ---------------------------------------------------------------------------

def test_dev_questions_parseable():
    import json
    path = pathlib.Path(__file__).parent.parent / "eval" / "questions" / "dev.jsonl"
    questions = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(questions) >= 10
    for q in questions:
        assert "id" in q
        assert "question" in q
        assert "type" in q
        assert "language" in q


def test_test_questions_parseable():
    import json
    path = pathlib.Path(__file__).parent.parent / "eval" / "questions" / "test.jsonl"
    questions = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(questions) >= 20


def test_configs_parseable():
    import json
    cfg_dir = pathlib.Path(__file__).parent.parent / "eval" / "configs"
    configs = list(cfg_dir.glob("*.json"))
    assert len(configs) == 4
    for cfg_path in configs:
        cfg = json.loads(cfg_path.read_text())
        assert "config_id" in cfg
        assert "retrieval_mode" in cfg
