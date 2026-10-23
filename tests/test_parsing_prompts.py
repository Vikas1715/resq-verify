import json

from backend.app.core.config import PROJECT_ROOT
from backend.app.schemas.plan import Plan
from backend.app.services.parsing import parse_json_object, parse_plan
from backend.app.services.prompts import PromptBook, build_feedback
from backend.app.validators.validator import validate_plan
from tests.helpers import broken_plan_dict, tiny_scenario, valid_plan_dict


def test_parse_plain_and_fenced_and_chatty_json():
    d = valid_plan_dict()
    for text in (json.dumps(d), "```json\n" + json.dumps(d) + "\n```", "Here you go:\n" + json.dumps(d) + "\nHope it helps {not json}"):
        r = parse_plan(text)
        assert r.plan is not None and r.error is None


def test_parse_failures_never_raise():
    for text in ("", "no json here", "{broken", "[1,2,3]", '{"vehicle_assignments": "oops"}', "null"):
        r = parse_plan(text)
        assert r.plan is None and r.error


def test_extra_keys_are_ignored_missing_lists_default():
    r = parse_plan(json.dumps({"vehicle_assignments": [], "mystery": 1}))
    assert isinstance(r.plan, Plan) and r.plan.team_assignments == []


def test_parse_json_object_handles_braces_in_strings():
    obj, err = parse_json_object('text {"a": "}{", "b": 1} tail')
    assert obj == {"a": "}{", "b": 1} and err is None


def test_prompt_versioning_and_hash(tmp_path):
    book = PromptBook(PROJECT_ROOT / "prompts")
    assert book.version and len(book.content_hash) == 12
    copy_dir = tmp_path / "p"
    copy_dir.mkdir()
    for f in (PROJECT_ROOT / "prompts").iterdir():
        (copy_dir / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    (copy_dir / "planner.txt").write_text((copy_dir / "planner.txt").read_text(encoding="utf-8") + "\nextra", encoding="utf-8")
    assert PromptBook(copy_dir).content_hash != book.content_hash


def test_feedback_structured_contains_exact_facts():
    sc = tiny_scenario()
    rep = validate_plan(sc, Plan.model_validate(broken_plan_dict()))
    fb = json.loads(build_feedback(rep, "structured"))
    assert fb["n_violations"] == rep.n_violations
    v = next(x for x in fb["violations"] if x["type"] == "vehicle_capacity")
    assert v["entity"] == "V1" and v["expected"] == 40 and v["actual"] == 45


def test_feedback_free_text_has_same_facts_without_json():
    sc = tiny_scenario()
    rep = validate_plan(sc, Plan.model_validate(broken_plan_dict()))
    txt = build_feedback(rep, "free_text")
    assert "V1" in txt and "40" in txt and not txt.lstrip().startswith("{")


def test_repair_prompt_contains_violations_but_critique_prompt_does_not():
    sc, book = tiny_scenario(), PromptBook(PROJECT_ROOT / "prompts")
    plan = Plan.model_validate(broken_plan_dict())
    rep = validate_plan(sc, plan)
    assert "vehicle_capacity" in book.repair(sc, plan, None, build_feedback(rep))
    crit = book.critique(sc, plan, None)
    assert "<<<VIOLATIONS>>>" not in crit and "vehicle_capacity" not in crit  # self-critique baseline never sees validator output
