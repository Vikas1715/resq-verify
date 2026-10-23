"""Prompt loading, rendering and versioning. Prompts live as plain text files in /prompts."""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.schemas.plan import Plan, plan_json_schema_text
from backend.app.schemas.scenario import Scenario
from backend.app.schemas.validation import ValidationReport
from backend.app.utils.ids import stable_hash

_FILES = ["_rules.txt", "planner.txt", "self_critique.txt", "revision.txt"]


class PromptBook:
    def __init__(self, prompt_dir: Path):
        self.dir = Path(prompt_dir)
        self.texts = {f: (self.dir / f).read_text(encoding="utf-8") for f in _FILES}
        version_file = self.dir / "VERSION"
        self.version = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else "unversioned"
        # the hash changes whenever any prompt text is edited, even if VERSION was not bumped
        self.content_hash = stable_hash(self.texts, 12)
        # per-file hashes: lets every run record show exactly which planner / critique / revision wording was used
        self.template_hashes = {f: stable_hash(t, 12) for f, t in self.texts.items()}

    def _render(self, name: str, **kw: str) -> str:
        text = self.texts[name].replace("{{rules}}", self.texts["_rules.txt"].strip())
        text = text.replace("{{plan_schema}}", plan_json_schema_text())
        for k, v in kw.items():
            text = text.replace("{{" + k + "}}", v)
        return text

    @staticmethod
    def _scenario_json(s: Scenario) -> str:
        return json.dumps(s.prompt_dict(), indent=1, ensure_ascii=False)

    @staticmethod
    def _plan_text(plan: Plan | None, raw: str | None) -> str:
        if plan is not None:
            return json.dumps(plan.model_dump(), indent=1, ensure_ascii=False)
        return (raw or "")[:6000]  # unparseable output is shown as-is (truncated)

    def planner(self, s: Scenario) -> str:
        return self._render("planner.txt", scenario_json=self._scenario_json(s))

    def critique(self, s: Scenario, plan: Plan | None, raw: str | None) -> str:
        return self._render("self_critique.txt", scenario_json=self._scenario_json(s), plan_json=self._plan_text(plan, raw))

    def revision(self, s: Scenario, plan: Plan | None, raw: str | None, feedback: str) -> str:
        """THE shared revision prompt used by M2 and M3. Only the {{FEEDBACK}} text differs between the two methods:
        M2 passes the LLM's own critique, M3 passes the deterministic validator's report. Nothing else may differ."""
        return self._render("revision.txt", scenario_json=self._scenario_json(s), plan_json=self._plan_text(plan, raw),
                            FEEDBACK=feedback)

    # thin aliases kept so older callers keep working; both go through the same template
    def revise(self, s: Scenario, plan: Plan | None, raw: str | None, critique_json: str) -> str:
        return self.revision(s, plan, raw, critique_json)

    def repair(self, s: Scenario, plan: Plan | None, raw: str | None, violations_block: str) -> str:
        return self.revision(s, plan, raw, violations_block)


def build_feedback(report: ValidationReport, fmt: str = "structured") -> str:
    """Turn a validation report into feedback text for the repair prompt.

    structured: machine-readable JSON (type, entity, expected, actual, message) grouped by rule
    free_text : the same facts as plain sentences (used to test whether feedback format matters)
    """
    if fmt == "free_text":
        lines = [f"- {v.message}" for v in report.violations]
        return f"{report.n_violations} problem(s) found:\n" + "\n".join(lines)
    payload = {
        "n_violations": report.n_violations,
        "counts_by_type": {k: n for k, n in report.counts_by_type.items() if n},
        "violations": [
            {"rule": v.rule, "type": v.type, "entity": v.entity, "incident_id": v.incident_id,
             "expected": v.expected, "actual": v.actual, "message": v.message}
            for v in report.violations
        ],
    }
    return json.dumps(payload, indent=1, ensure_ascii=False)
