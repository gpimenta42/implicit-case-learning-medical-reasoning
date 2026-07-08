from __future__ import annotations

from collections import defaultdict
from typing import Any


VALID_OPTIONS = ["A", "B", "C", "D", "E"]


ANSWER_SCHEMA = {
    "name": "base_reasoning_trace_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Concise step-by-step clinical reasoning anchored to the stem and answer options.",
            },
            "selected_answer": {
                "type": "string",
                "enum": VALID_OPTIONS,
                "description": "Single best answer letter.",
            },
        },
        "required": ["reasoning", "selected_answer"],
    },
}


SOLVER_SCHEMA = {
    "name": "icl_edr_solver_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Concise step-by-step clinical reasoning for the new target question.",
            },
            "answer_idx": {
                "type": "string",
                "enum": VALID_OPTIONS,
                "description": "One displayed option letter for the new target question.",
            },
        },
        "required": ["reasoning", "answer_idx"],
    },
}


JUDGE_SCHEMA = {
    "name": "icl_edr_grouped_judge_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "option": {"type": "string", "enum": VALID_OPTIONS},
                        "answer": {"type": "string"},
                        "case_for_candidate": {
                            "type": "string",
                            "description": "Why this candidate could be correct based primarily on the original stem, answer options, and exact task being asked, using grouped traces only as supporting or contrasting evidence.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["favored", "plausible", "rejected", "unclear"],
                        },
                    },
                    "required": ["option", "answer", "case_for_candidate", "status"],
                },
            },
            "decisive_distinction": {
                "type": "string",
                "description": "The main stem-grounded distinction and task framing that determines the final answer.",
            },
            "selected_answer": {"type": "string", "enum": VALID_OPTIONS},
            "selected_answer_text": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": [
            "candidate_assessments",
            "decisive_distinction",
            "selected_answer",
            "selected_answer_text",
            "confidence",
        ],
    },
}


JUDGE_SYSTEM_PROMPT = """You are the final answer judge for a medical multiple-choice ensemble repair experiment.
You must choose one answer from the candidate options provided.
You will receive the original question/options and raw target reasoning traces, grouped by the answer selected by each rollout.
The number of traces in a group reflects how many rollouts selected that candidate, but the majority vote may be wrong.
Use the original stem and answer options as the source of truth.
Evaluate which candidate is best supported by the stem after comparing the grouped reasoning traces.
Do not choose an answer solely because it has more traces.
Do not choose an answer outside the candidate options allowed.
Return valid JSON only."""


def clean_letter(value: Any, valid_options: list[str] | None = None) -> str | None:
    valid = set(valid_options or VALID_OPTIONS)
    if value is None:
        return None
    text = str(value).strip().upper()
    text = text.strip('"').strip("'").strip("`.,:;()[]{}")
    if text in valid:
        return text
    import re

    match = re.search(r"\b([A-E])\b", text)
    if match and match.group(1) in valid:
        return match.group(1)
    return None


def setting_control_system_prompt(setting: Any) -> str | None:
    text = setting.system_prefix.strip()
    return text or None


def icl_judge_system_prompt(setting: Any) -> str:
    prefix = setting.system_prefix.strip()
    return f"{prefix}\n\n{JUDGE_SYSTEM_PROMPT}".strip() if prefix else JUDGE_SYSTEM_PROMPT


def format_options(options: dict[str, Any]) -> str:
    if not options:
        return ""
    return "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))


def format_exemplars(exemplars: list[dict[str, Any]], exemplar_mode: str = "gold") -> str:
    blocks = []
    for idx, ex in enumerate(exemplars, start=1):
        options = ex.get("options") or {}
        option_text = format_options(options)
        gold = ex.get("correct_answer_idx") or ex.get("gold_answer_idx")
        gold_text = ex.get("correct_answer_text") or ex.get("gold_answer_text") or ""
        rationale_block = ""
        if exemplar_mode == "gold_rationale":
            rationale = str(ex.get("gold_rationale") or "").strip()
            rationale_block = f"\n\nGenerated explanation for gold answer:\n{rationale}" if rationale else ""
        blocks.append(
            f"""Example {idx}:
Question:
{ex.get("question", "")}

Options:
{option_text}

Gold answer: {gold}. {gold_text}{rationale_block}"""
        )
    return "\n\n".join(blocks)


def base_system_prompt(setting: Any) -> str:
    prefix = setting.system_prefix.strip()
    body = """You are a medical expert answering a multiple-choice medical exam question.

Use only the question stem, answer options, and foundational medical knowledge.
If the question is written in Portuguese, reason from the Portuguese text without translating away clinically meaningful wording.

Instructions:
- First identify what the question is asking: diagnosis / etiology / confirmatory test / management-next-step / association.
- Keep your final answer aligned to that task.
- Reason briefly and concretely from the stem.
- Eliminate answer choices contradicted by the vignette.

Return structured fields:
- reasoning: concise step-by-step clinical reasoning
- selected_answer: one of A/B/C/D/E"""
    return f"{prefix}\n\n{body}".strip() if prefix else body


def cot_user_prompt(row: Any, setting: Any, rollout_idx: int) -> str:
    prefix = f"{setting.user_prefix.strip()}\n\n" if setting.user_prefix.strip() else ""
    return f"""{prefix}Question:
{row["question_with_options"]}

Return only JSON matching the requested schema.""".strip()


def icl_solver_user_prompt(
    row: Any,
    exemplars: list[dict[str, Any]],
    setting: Any,
    solver_idx: int,
    ensemble_repeat_idx: int,
    exemplar_mode: str = "gold",
) -> str:
    prefix = f"{setting.user_prefix.strip()}\n\n" if setting.user_prefix.strip() else ""
    examples = format_exemplars(exemplars, exemplar_mode=exemplar_mode) if exemplars else "No retrieved source examples were available. Solve from the target stem and options only."
    if exemplar_mode == "gold_rationale":
        example_instruction = "Each example includes the known gold answer and a generated explanation for that known answer."
    else:
        example_instruction = "Each example includes the known gold answer but no generated explanation."
    target_question = str(row.get("question_stem") or row.get("question") or row.get("question_with_options") or "").strip()
    displayed_options = format_options(row.get("options") or {})
    return f"""{prefix}You are answering a medical multiple-choice exam question.

Use the retrieved examples only as similar solved cases. {example_instruction}
Do not force transfer when the target stem differs. Reason from the target stem, answer options, and foundational medical knowledge.
Choose exactly one displayed option letter for the new target question.

Retrieved gold-answer examples:

{examples}

New target question:
{target_question}

Displayed target options:
{displayed_options}

Return JSON matching the enforced schema with:
- reasoning: concise step-by-step reasoning for the new target question
- answer_idx: one displayed option letter for the new target question""".strip()


def judge_user_prompt(
    row: Any,
    solver_rows: list[dict[str, Any]],
    setting: Any,
    ensemble_repeat_idx: int,
    judge_repeat_idx: int,
) -> str:
    prefix = f"{setting.user_prefix.strip()}\n\n" if setting.user_prefix.strip() else ""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for solver_row in solver_rows:
        answer = clean_letter(solver_row.get("selected_answer"), row.get("valid_options") or VALID_OPTIONS)
        if answer:
            grouped[answer].append(solver_row)

    candidate_blocks = []
    options = row.get("options") or {}
    for answer, rows in sorted(grouped.items()):
        traces = "\n\n".join(
            "Trace {trace_idx} for candidate {answer}:\n{reasoning_trace}".format(
                trace_idx=idx,
                answer=answer,
                reasoning_trace=str(r.get("reasoning") or "").strip(),
            )
            for idx, r in enumerate(rows, start=1)
        )
        candidate_blocks.append(
            f"""Candidate {answer}. {options.get(answer, "")}
Number of target traces selecting this candidate: {len(rows)}
Grouped reasoning traces:
{traces}"""
        )
    allowed = ", ".join(f"{answer}. {options.get(answer, '')}" for answer in sorted(grouped))

    return f"""{prefix}Original question and answer options.

Question and answer options:
{row["question_with_options"]}

Candidate options allowed for final selection:
{allowed}

Grouped target reasoning traces by selected answer:
The following section shows all raw reasoning traces from this solver ensemble, grouped by the answer each rollout selected. The group with the most traces is the majority vote, but the majority vote may be wrong.

{chr(10).join(candidate_blocks)}

Task:
1. Assess each candidate option using the original stem/options and the grouped reasoning traces for that candidate.
2. Identify the decisive distinction that separates the candidates for this question.
3. Choose exactly one final answer from the candidate options allowed above.

Rules:
- Do not choose an option outside the candidate options allowed above.
- Treat the original question as authoritative if a trace conflicts with it.
- Do not use confidence, verbosity, or trace length as evidence of correctness.
- Do not choose solely because one candidate has more traces; the majority vote may be wrong.

Return JSON matching the requested schema, with fields:
- candidate_assessments
- decisive_distinction
- selected_answer: one of A/B/C/D/E
- selected_answer_text
- confidence: low, medium, or high""".strip()
