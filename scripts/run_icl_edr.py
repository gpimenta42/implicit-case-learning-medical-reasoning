#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "icl_edr_runner.py").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find repository root containing src/icl_edr_runner.py and README.md.")


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from icl_edr_runner import ICLEDRRunner, ModelSetting, load_env  # noqa: E402


class Style:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stdout.isatty()

    def apply(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def heading(self, text: str) -> str:
        return self.apply(text, "1;36")

    def label(self, text: str) -> str:
        return self.apply(text, "1")

    def good(self, text: str) -> str:
        return self.apply(text, "32")

    def warn(self, text: str) -> str:
        return self.apply(text, "33")

    def path(self, text: str) -> str:
        return self.apply(text, "36")

    def target(self, text: str) -> str:
        return self.apply(text, "96")

    def example(self, text: str) -> str:
        return self.apply(text, "94")

    def reasoning(self, text: str) -> str:
        return self.apply(text, "95")

    def judge(self, text: str) -> str:
        return self.apply(text, "93")


def resolve_path(value: str | None, default: Path | None = None) -> Path | None:
    raw = value if value not in {None, ""} else default
    if raw is None:
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_json_arg(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--extra-body must decode to a JSON object.")
    return parsed


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def format_float(value: str | float | None, *, digits: int = 1) -> str:
    if value in {None, ""}:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def truncate_text(text: Any, limit: int = 360) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def format_options(options: Any) -> str:
    if not isinstance(options, dict):
        return ""
    rows = []
    for key in sorted(options):
        rows.append(f"{key}. {options[key]}")
    return "\n".join(rows)


def indent_block(text: Any, prefix: str = "    ") -> str:
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in str(text or "").splitlines())


def print_colored_block(label: str, text: Any, *, style: Style, color: str = "target") -> None:
    print(f"  {style.label(label)}")
    block = indent_block(text)
    color_fn = getattr(style, color)
    print(color_fn(block))


def print_question_details(runner: ICLEDRRunner, *, limit: int, style: Style) -> None:
    if limit <= 0:
        return
    per_question_rows = read_csv_rows(runner.per_question_csv)
    raw_rows = read_jsonl_rows(runner.outputs_jsonl)
    if not per_question_rows or not raw_rows:
        return

    print()
    print(style.heading(f"Details (first {min(limit, len(per_question_rows))} question(s))"))
    for question_idx, result in enumerate(per_question_rows[:limit], 1):
        qkey = result.get("question_key", "")
        related_rows = [row for row in raw_rows if str(row.get("question_key", "")) == qkey]
        target = related_rows[0] if related_rows else {}
        print()
        print(style.label(f"Question {question_idx}: {qkey or 'unknown'}"))
        print(f"  Dataset: {result.get('dataset_name', 'n/a')} | difficulty: {result.get('difficulty', 'n/a')}")
        print(f"  Gold: {result.get('ground_truth') or 'n/a'} | selected: {result.get('selected_answer') or 'n/a'} | correct: {result.get('is_correct') or 'n/a'}")
        print(f"  Final source: {result.get('final_source') or 'n/a'} | judge ran: {result.get('judge_ran') or 'False'}")

        question_text = target.get("question_with_options") or target.get("question")
        if question_text:
            print_colored_block("Target question:", question_text, style=style, color="target")

        exemplars = target.get("exemplars") or []
        if exemplars:
            print(f"  {style.label('Retrieved cases:')}")
            for exemplar in exemplars:
                rank = exemplar.get("rank", "?")
                source_key = exemplar.get("question_key") or exemplar.get("source_case_id") or "unknown"
                answer = exemplar.get("correct_answer_idx") or "?"
                answer_text = exemplar.get("correct_answer_text") or ""
                similarity = exemplar.get("dense_similarity")
                similarity_text = f", sim={float(similarity):.3f}" if isinstance(similarity, (int, float)) else ""
                example_text = (
                    f"{rank}. {source_key}{similarity_text}\n"
                    f"Question:\n{exemplar.get('question', '')}\n\n"
                    f"Options:\n{format_options(exemplar.get('options'))}\n\n"
                    f"Gold answer: {answer}. {answer_text}".rstrip()
                )
                print(style.example(indent_block(example_text)))

        solver_rows = [row for row in related_rows if row.get("stage") == "icl_solver"]
        if solver_rows:
            selected_answers = [row.get("selected_answer") or "n/a" for row in solver_rows]
            unique_answers = []
            for answer in selected_answers:
                if answer not in unique_answers:
                    unique_answers.append(answer)

            if len(unique_answers) == 1:
                solver = solver_rows[0]
                label = f"Solver reasoning (ensemble agreed on {unique_answers[0]}):"
                reasoning_text = f"solver {solver.get('solver_idx', '?')}: {solver.get('reasoning') or 'n/a'}"
                print_colored_block(label, reasoning_text, style=style, color="reasoning")
            else:
                print(f"  {style.label('Solver reasoning by selected option:')}")
                for answer in unique_answers:
                    solver = next(row for row in solver_rows if (row.get("selected_answer") or "n/a") == answer)
                    reasoning_text = f"selected {answer}, solver {solver.get('solver_idx', '?')}: {solver.get('reasoning') or 'n/a'}"
                    print(style.reasoning(indent_block(reasoning_text)))

        judge_rows = [row for row in related_rows if row.get("stage") == "icl_judge"]
        if judge_rows:
            judge = judge_rows[0]
            judge_text = (
                f"selected {judge.get('selected_answer') or 'n/a'}"
                f"; reasoning: {judge.get('reasoning') or judge.get('raw_response') or 'n/a'}"
            )
            print_colored_block("Judge decision:", judge_text, style=style, color="judge")


def print_cli_summary(runner: ICLEDRRunner, *, run_api: bool, show_details: bool, show_details_limit: int, no_color: bool) -> None:
    style = Style(enabled=not no_color)
    summary_rows = read_csv_rows(runner.summary_csv)
    per_question_rows = read_csv_rows(runner.per_question_csv)

    print()
    if run_api:
        print(style.heading("ICL-EDR run complete"))
    elif summary_rows:
        print(style.heading("ICL-EDR summary from existing outputs"))
    else:
        print(style.heading("ICL-EDR dry run complete"))
    print(f"{style.label('Run name:')} {runner.run_name}")
    print(f"{style.label('Output:')} {style.path(display_path(runner.out_dir))}")

    if summary_rows:
        print()
        print(style.label("Results:"))
        for row in summary_rows:
            accuracy_pct = row.get("accuracy_pct") or (float(row.get("accuracy", 0.0)) * 100 if row.get("accuracy") else "")
            accuracy_text = style.good(f"{format_float(accuracy_pct)}%")
            print(
                "  "
                f"{row.get('method', 'method')} | "
                f"{row.get('dataset_name', 'dataset')} | "
                f"n={row.get('n', '0')} | "
                f"accuracy={accuracy_text} | "
                f"tokens/q={format_float(row.get('mean_total_tokens'), digits=0)} | "
                f"API calls/q={format_float(row.get('mean_api_calls'))} | "
                f"judge rate={format_float(float(row.get('judge_rate', 0.0)) * 100 if row.get('judge_rate') else '', digits=1)}%"
            )
    else:
        print()
        print(f"{style.label('Results:')} no completed per-question rows yet.")
        if not run_api:
            print(style.warn("No provider calls were made. Re-run with --run-api to execute the model calls."))

    if per_question_rows:
        first = per_question_rows[0]
        selected = first.get("selected_answer") or "n/a"
        gold = first.get("ground_truth") or "n/a"
        correct = first.get("is_correct") or "n/a"
        correct_text = style.good(correct) if str(correct).lower() == "true" else style.warn(correct)
        print()
        print(f"{style.label('First question:')} selected={selected}, gold={gold}, correct={correct_text}")

    if show_details:
        print_question_details(runner, limit=show_details_limit, style=style)

    print()
    print(style.label("Files:"))
    print(f"  Summary: {style.path(display_path(runner.summary_csv))}")
    print(f"  Per-question results: {style.path(display_path(runner.per_question_csv))}")
    print(f"  Raw model outputs: {style.path(display_path(runner.outputs_jsonl))}")
    print(f"  Prompt/schema snapshot: {style.path(display_path(runner.prompt_md))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the clean ICL-EDR reference implementation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-csv", default=os.getenv("ICL_EDR_TARGET_CSV"), help="Target-question CSV.")
    parser.add_argument("--retrieval-json", default=os.getenv("ICL_EDR_RETRIEVAL_JSON"), help="ICL-EDR retrieval JSON.")
    parser.add_argument("--source-csv", default=os.getenv("ICL_EDR_SOURCE_CSV"), help="Labelled source-case CSV used with --build-retrieval.")
    parser.add_argument("--output-root", default=None, help="Directory for run outputs.")
    parser.add_argument("--run-name", default="cli_icl_edr_run", help="Name of the output subdirectory.")

    parser.add_argument("--provider", default="openai", choices=["openai", "huggingface_inference_providers"], help="Model provider branch.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"), help="Provider model identifier.")
    parser.add_argument("--hf-provider", default=os.getenv("HF_PROVIDER", ""), help="Hugging Face Inference Provider name.")
    parser.add_argument("--setting-name", default="cli_setting", help="Name stored in result tables.")
    parser.add_argument("--reasoning-effort", default=os.getenv("OPENAI_REASONING_EFFORT", "medium"), help="Reasoning effort for OpenAI-style calls. Use empty string to omit.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional top-p value.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum output tokens.")
    parser.add_argument("--extra-body", type=parse_json_arg, default=None, help="Optional JSON object passed to provider-specific extra_body.")
    parser.add_argument("--no-response-format", action="store_true", help="Disable structured response_format enforcement.")

    parser.add_argument("--include-cot-baseline", action="store_true", help="Also run the matched CoT baseline.")
    parser.add_argument("--cot-rollouts", type=int, default=1, help="CoT completions per question when --include-cot-baseline is used.")
    parser.add_argument("--retrieval-k", type=int, default=2, help="Number of retrieved cases shown to ICL-EDR.")
    parser.add_argument("--solver-count", type=int, default=3, help="ICL-EDR solver ensemble size.")
    parser.add_argument("--ensemble-repeats", type=int, default=1, help="Independent ICL-EDR ensemble repeats per question.")
    parser.add_argument("--judge-repeats", type=int, default=1, help="Routed judge repeats for mixed ensembles.")
    parser.add_argument("--no-batch-solver-choices", action="store_true", help="Request solver completions as separate calls instead of one n=S call.")

    parser.add_argument("--run-api", action="store_true", help="Execute provider calls. By default the script performs a dry run.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel task workers.")
    parser.add_argument("--max-tasks", type=int, default=20, help="Safety cap for tasks in this run. Use 0 for no cap.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore existing successful outputs and rerun tasks.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Provider request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum provider retries per task.")
    parser.add_argument("--show-details", action="store_true", help="Print target question, retrieved cases, answers, and solver reasoning for a small number of completed rows.")
    parser.add_argument("--show-details-limit", type=int, default=1, help="Number of completed rows shown when --show-details is used.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color in the final CLI summary.")
    parser.add_argument("--verbose-runner-output", action="store_true", help="Show the lower-level runner progress and pandas summary output.")

    parser.add_argument("--build-retrieval", action="store_true", help="Build the retrieval JSON before running ICL-EDR.")
    parser.add_argument("--build-top-k", type=int, default=4, help="Nearest neighbours stored per target when building retrieval.")
    parser.add_argument("--embedding-model", default=os.getenv("ICL_EDR_EMBEDDING_MODEL", "text-embedding-3-large"), help="Embedding model for --build-retrieval.")
    parser.add_argument("--source-pool-mode", default=os.getenv("ICL_EDR_SOURCE_POOL_MODE", "custom_source_pool"), help="Metadata label stored in built retrieval entries.")
    parser.add_argument("--openai-embedding-batch-size", type=int, default=128, help="Embedding batch size for --build-retrieval.")
    return parser


def build_retrieval_if_requested(args: argparse.Namespace, target_csv: Path, retrieval_json: Path) -> None:
    if not args.build_retrieval:
        return
    source_csv = resolve_path(args.source_csv, PROJECT_ROOT / "example_data" / "dummy_source_cases.csv")
    if source_csv is None:
        raise RuntimeError("--source-csv is required when --build-retrieval is used.")
    if not source_csv.exists():
        raise FileNotFoundError(f"Source-case CSV not found: {source_csv}")

    from retrieval_builder import build_retrieval_json

    build_retrieval_json(
        target_csv_path=target_csv,
        source_csv_path=source_csv,
        output_json_path=retrieval_json,
        top_k=args.build_top_k,
        embedding_model=args.embedding_model,
        source_pool_mode=args.source_pool_mode,
        openai_batch_size=args.openai_embedding_batch_size,
    )
    print("Wrote retrieval JSON:", retrieval_json)


def main() -> None:
    load_env(PROJECT_ROOT)
    args = build_parser().parse_args()

    target_csv = resolve_path(args.target_csv, PROJECT_ROOT / "example_data" / "dummy_target_question.csv")
    retrieval_json = resolve_path(args.retrieval_json, PROJECT_ROOT / "example_data" / "dummy_retrieval.json")
    output_root = resolve_path(args.output_root, PROJECT_ROOT / "results" / "run_outputs")
    if target_csv is None or retrieval_json is None or output_root is None:
        raise RuntimeError("Target CSV, retrieval JSON, and output root must resolve to paths.")
    if not target_csv.exists():
        raise FileNotFoundError(f"Target-question CSV not found: {target_csv}")

    build_retrieval_if_requested(args, target_csv, retrieval_json)
    if not retrieval_json.exists():
        raise FileNotFoundError(f"Retrieval JSON not found: {retrieval_json}. Use --build-retrieval or provide --retrieval-json.")

    reasoning_effort = args.reasoning_effort.strip() if isinstance(args.reasoning_effort, str) else args.reasoning_effort
    setting = ModelSetting(
        name=args.setting_name,
        provider=args.provider,
        model_id=args.model,
        hf_provider=args.hf_provider,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        reasoning_effort=reasoning_effort or None,
        extra_body=args.extra_body,
        use_response_format=not args.no_response_format,
        notes="CLI ICL-EDR run.",
    )

    runner = ICLEDRRunner(
        run_name=args.run_name,
        settings=[setting],
        project_root=PROJECT_ROOT,
        output_root=output_root,
        target_csv_path=target_csv,
        retrieval_json_path=retrieval_json,
        include_cot_baseline=args.include_cot_baseline,
        cot_rollouts=args.cot_rollouts,
        retrieval_k=args.retrieval_k,
        icl_solver_count=args.solver_count,
        icl_ensemble_repeats=args.ensemble_repeats,
        icl_judge_repeats=args.judge_repeats,
        batch_solver_choices=not args.no_batch_solver_choices,
        run_api_calls=args.run_api,
        workers=args.workers,
        max_tasks_this_run=args.max_tasks,
        force_rerun=args.force_rerun,
        request_timeout=args.timeout,
        max_retries=args.max_retries,
    )
    if args.verbose_runner_output:
        runner.run()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run()
    print_cli_summary(
        runner,
        run_api=args.run_api,
        show_details=args.show_details,
        show_details_limit=args.show_details_limit,
        no_color=args.no_color,
    )


if __name__ == "__main__":
    main()
