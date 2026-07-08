from __future__ import annotations

import concurrent.futures as futures
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from .prompts import (
        ANSWER_SCHEMA,
        JUDGE_SCHEMA,
        SOLVER_SCHEMA,
        VALID_OPTIONS,
        base_system_prompt,
        clean_letter,
        cot_user_prompt,
        icl_judge_system_prompt,
        icl_solver_user_prompt,
        judge_user_prompt,
        setting_control_system_prompt,
    )
except ImportError:
    from prompts import (
        ANSWER_SCHEMA,
        JUDGE_SCHEMA,
        SOLVER_SCHEMA,
        VALID_OPTIONS,
        base_system_prompt,
        clean_letter,
        cot_user_prompt,
        icl_judge_system_prompt,
        icl_solver_user_prompt,
        judge_user_prompt,
        setting_control_system_prompt,
    )

TARGET_CSV_REL = Path("target_questions.csv")
RETRIEVAL_JSON_REL = Path("retrieval_cases.json")
DEFAULT_OUTPUT_ROOT_REL = Path("results/run_outputs")


@dataclass(frozen=True)
class ModelSetting:
    name: str
    provider: str
    model_id: str
    temperature: float
    max_tokens: int
    hf_provider: str = ""
    reasoning_effort: str | None = None
    cot_temperature: float | None = None
    icl_temperature: float | None = None
    top_p: float | None = None
    extra_body: dict[str, Any] | None = None
    use_response_format: bool = True
    system_prefix: str = ""
    user_prefix: str = ""
    notes: str = ""


def find_project_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src" / "icl_edr_runner.py").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find project root containing src/icl_edr_runner.py and README.md.")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env(project_root: Path) -> None:
    for path in [project_root / ".env", Path.home() / ".env"]:
        load_env_file(path)


def get_hf_token(project_root: Path) -> str | None:
    load_env(project_root)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def get_openai_api_key(project_root: Path) -> str | None:
    load_env(project_root)
    return os.getenv("OPENAI_API_KEY")


def parse_jsonish(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def parse_valid_options(value: Any) -> list[str]:
    parsed = parse_jsonish(value, default=None)
    if isinstance(parsed, list):
        opts = [str(x).strip().upper() for x in parsed if str(x).strip().upper() in VALID_OPTIONS]
        return opts or VALID_OPTIONS
    return VALID_OPTIONS


def parse_options_from_question(question_with_options: str) -> tuple[str, dict[str, str]]:
    text = str(question_with_options or "").strip()
    matches = list(re.finditer(r"(?:^|\n)\(([A-E])\)\s*", text))
    if not matches:
        return text, {}
    stem = text[: matches[0].start()].strip()
    options: dict[str, str] = {}
    for idx, match in enumerate(matches):
        letter = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        options[letter] = text[start:end].strip()
    return stem, options


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def to_plain(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    return obj


def usage_value(usage: Any, key: str) -> int | None:
    plain = to_plain(usage) or {}
    if not isinstance(plain, dict):
        return None
    value = plain.get(key)
    if value is None and key == "reasoning_tokens":
        details = plain.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            value = details.get("reasoning_tokens")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_usage(usage: Any) -> dict[str, Any]:
    plain = to_plain(usage) or {}
    if not isinstance(plain, dict):
        plain = {}
    return {
        **plain,
        "prompt_tokens": usage_value(plain, "prompt_tokens"),
        "completion_tokens": usage_value(plain, "completion_tokens"),
        "total_tokens": usage_value(plain, "total_tokens"),
        "reasoning_tokens": usage_value(plain, "reasoning_tokens"),
    }


def split_int(total: int | None, n: int) -> list[int | None]:
    if total is None:
        return [None] * n
    base = int(total) // n
    remainder = int(total) % n
    return [base + (1 if i < remainder else 0) for i in range(n)]


def strip_thinking_blocks(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    parts = re.findall(r"<think>(.*?)</think>", raw, flags=re.S | re.I)
    thinking = "\n\n".join(part.strip() for part in parts if part.strip())
    without = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).strip()
    return thinking, without


def extract_json_obj(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    _, raw = strip_thinking_blocks(raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def majority_vote(answers: list[str | None]) -> tuple[str | None, int, dict[str, int]]:
    counts = Counter(a for a in answers if a)
    if not counts:
        return None, 0, {}
    max_count = max(counts.values())
    winners = sorted([answer for answer, count in counts.items() if count == max_count])
    return winners[0], max_count, dict(sorted(counts.items()))


def safe_key(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(text))


def load_target_questions(project_root: Path, target_csv_path: Path | None = None) -> pd.DataFrame:
    path = target_csv_path or (project_root / TARGET_CSV_REL)
    df = pd.read_csv(path)
    df = df[df["dataset_name"].isin(["medqa_test", "pna_all"])].copy()
    df["valid_options"] = df["valid_options"].apply(parse_valid_options)
    parsed = df["question_with_options"].apply(parse_options_from_question)
    df["question_stem"] = [x[0] for x in parsed]
    df["options"] = [x[1] for x in parsed]
    return df.reset_index(drop=True)


def resolve_path(project_root: Path, path_text: str | os.PathLike[str]) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_retrieval(project_root: Path, retrieval_json_path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    path = retrieval_json_path or (project_root / RETRIEVAL_JSON_REL)
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_exemplar_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sid = str(row.get("source_case_id") or (row.get("source") or {}).get("source_case_id") or "").strip()
        rationale = str(row.get("gold_rationale") or row.get("generated_reasoning") or "").strip()
        if sid and row.get("is_valid") and rationale:
            cache[sid] = row
    return cache


def exemplars_for(row: pd.Series | dict[str, Any], retrieval: dict[str, list[dict[str, Any]]], k: int = 2) -> list[dict[str, Any]]:
    keys = [
        str(row.get("question_key", "")),
        str(row.get("canonical_question_key", "")),
    ]
    for key in keys:
        if key and key in retrieval:
            return list(retrieval[key])[:k]
    return []


def attach_gold_rationales(exemplars: list[dict[str, Any]], source_exemplar_cache: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for ex in exemplars:
        sid = str(ex.get("source_case_id") or "").strip()
        cached = source_exemplar_cache.get(sid, {})
        rationale = str(cached.get("gold_rationale") or cached.get("generated_reasoning") or "").strip()
        enriched_ex = dict(ex)
        if rationale:
            enriched_ex["gold_rationale"] = rationale
        enriched.append(enriched_ex)
    return enriched


class ICLEDRRunner:
    """Run ICL-EDR, optionally with a matched CoT baseline.

    Set include_cot_baseline=True when you want the runner to schedule CoT
    tasks alongside ICL-EDR tasks for direct per-question comparison.
    """

    def __init__(
        self,
        *,
        run_name: str,
        settings: list[ModelSetting],
        project_root: Path | None = None,
        output_root: Path | None = None,
        env_prefix: str = "ICL_EDR",
        include_cot_baseline: bool = True,
        cot_rollouts: int = 1,
        icl_ensemble_repeats: int = 1,
        icl_solver_count: int = 3,
        icl_judge_repeats: int = 1,
        batch_solver_choices: bool = True,
        retrieval_k: int = 2,
        target_csv_path: Path | None = None,
        retrieval_json_path: Path | None = None,
        exemplar_mode: str = "gold",
        source_exemplar_cache_path: Path | None = None,
        run_api_calls: bool = False,
        workers: int = 8,
        batch_task_count: int = 100,
        max_tasks_this_run: int = 0,
        force_rerun: bool = False,
        request_timeout: float = 180.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        stop_on_quota_error: bool = True,
    ) -> None:
        self.project_root = project_root or find_project_root()
        self.run_name = run_name
        self.settings = settings
        self.env_prefix = env_prefix
        self.output_root = output_root or (self.project_root / DEFAULT_OUTPUT_ROOT_REL)
        self.out_dir = self.output_root / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_jsonl = self.out_dir / "outputs.jsonl"
        self.per_question_csv = self.out_dir / "per_question_results.csv"
        self.summary_csv = self.out_dir / "summary_by_setting_method_dataset.csv"
        self.summary_json = self.out_dir / "summary.json"
        self.prompt_md = self.out_dir / "PROMPT_AND_SCHEMA.md"

        self.include_cot_baseline = include_cot_baseline
        self.cot_rollouts = cot_rollouts
        self.icl_ensemble_repeats = icl_ensemble_repeats
        self.icl_solver_count = icl_solver_count
        self.icl_judge_repeats = icl_judge_repeats
        self.batch_solver_choices = batch_solver_choices
        self.retrieval_k = retrieval_k
        self.target_csv_path = target_csv_path or (self.project_root / TARGET_CSV_REL)
        self.retrieval_json_path = retrieval_json_path or (self.project_root / RETRIEVAL_JSON_REL)
        self.exemplar_mode = exemplar_mode
        if self.exemplar_mode not in {"gold", "gold_rationale"}:
            raise ValueError(f"Unsupported exemplar_mode: {self.exemplar_mode}")
        self.source_exemplar_cache_path = source_exemplar_cache_path
        self.run_api_calls = run_api_calls
        self.workers = workers
        self.batch_task_count = batch_task_count
        self.max_tasks_this_run = max_tasks_this_run
        self.force_rerun = force_rerun
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.stop_on_quota_error = stop_on_quota_error

        self.target_df = load_target_questions(self.project_root, self.target_csv_path)
        self.retrieval = load_retrieval(self.project_root, self.retrieval_json_path)
        self.source_exemplar_cache = load_source_exemplar_cache(self.source_exemplar_cache_path)
        if self.exemplar_mode == "gold_rationale" and not self.source_exemplar_cache:
            raise RuntimeError("gold_rationale exemplar mode requires a non-empty source exemplar cache.")
        self.thread_local = threading.local()

    @classmethod
    def from_env(cls, *, run_name: str, settings: list[ModelSetting], env_prefix: str = "ICL_EDR") -> "ICLEDRRunner":
        project_root = find_project_root()
        load_env(project_root)
        target_csv = os.getenv(f"{env_prefix}_TARGET_CSV")
        retrieval_json = os.getenv(f"{env_prefix}_RETRIEVAL_JSON")
        source_exemplar_cache = os.getenv(f"{env_prefix}_SOURCE_EXEMPLAR_CACHE")
        return cls(
            run_name=run_name,
            settings=settings,
            project_root=project_root,
            env_prefix=env_prefix,
            include_cot_baseline=os.getenv(f"{env_prefix}_INCLUDE_COT_BASELINE", "1") == "1",
            cot_rollouts=int(os.getenv(f"{env_prefix}_COT_ROLLOUTS", "1")),
            icl_ensemble_repeats=int(os.getenv(f"{env_prefix}_ICL_ENSEMBLE_REPEATS", "1")),
            icl_solver_count=int(os.getenv(f"{env_prefix}_ICL_SOLVER_COUNT", "3")),
            icl_judge_repeats=int(os.getenv(f"{env_prefix}_ICL_JUDGE_REPEATS", "1")),
            batch_solver_choices=os.getenv(f"{env_prefix}_BATCH_SOLVER_CHOICES", "1") == "1",
            retrieval_k=int(os.getenv(f"{env_prefix}_RETRIEVAL_K", "2")),
            target_csv_path=resolve_path(project_root, target_csv) if target_csv else None,
            retrieval_json_path=resolve_path(project_root, retrieval_json) if retrieval_json else None,
            exemplar_mode=os.getenv(f"{env_prefix}_EXEMPLAR_MODE", "gold"),
            source_exemplar_cache_path=resolve_path(project_root, source_exemplar_cache) if source_exemplar_cache else None,
            run_api_calls=os.getenv(f"RUN_{env_prefix}", "0") == "1",
            workers=int(os.getenv(f"{env_prefix}_WORKERS", "8")),
            batch_task_count=int(os.getenv(f"{env_prefix}_BATCH_TASK_COUNT", "100")),
            max_tasks_this_run=int(os.getenv(f"{env_prefix}_MAX_TASKS", "0")),
            force_rerun=os.getenv(f"{env_prefix}_FORCE_RERUN", "0") == "1",
            request_timeout=float(os.getenv(f"{env_prefix}_TIMEOUT", "180")),
            max_retries=int(os.getenv(f"{env_prefix}_MAX_RETRIES", "3")),
            retry_backoff_seconds=float(os.getenv(f"{env_prefix}_RETRY_BACKOFF_SECONDS", "2")),
        )

    def write_prompt_snapshot(self) -> None:
        payload = {
            "run_name": self.run_name,
            "settings": [asdict(s) for s in self.settings],
            "include_cot_baseline": self.include_cot_baseline,
            "cot_rollouts": self.cot_rollouts,
            "icl_ensemble_repeats": self.icl_ensemble_repeats,
            "icl_solver_count": self.icl_solver_count,
            "icl_judge_repeats": self.icl_judge_repeats,
            "batch_solver_choices": self.batch_solver_choices,
            "retrieval_k": self.retrieval_k,
            "exemplar_mode": self.exemplar_mode,
            "source_exemplar_cache": None if self.source_exemplar_cache_path is None else str(self.source_exemplar_cache_path.resolve()),
            "target_csv": str(self.target_csv_path.resolve()),
            "retrieval_json": str(self.retrieval_json_path.resolve()),
        }
        cot_section = ""
        if self.include_cot_baseline:
            cot_section = (
                "\n## CoT system prompt\n\n"
                f"```text\n{base_system_prompt(self.settings[0])}\n```\n\n"
            )
        self.prompt_md.write_text(
            "# ICL-EDR runner\n\n"
            f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n\n"
            f"{cot_section}"
            "## ICL-EDR solver schema\n\n"
            f"```json\n{json.dumps(SOLVER_SCHEMA, ensure_ascii=False, indent=2)}\n```\n\n"
            "## ICL-EDR judge schema\n\n"
            f"```json\n{json.dumps(JUDGE_SCHEMA, ensure_ascii=False, indent=2)}\n```\n",
            encoding="utf-8",
        )

    def icl_method_name(self) -> str:
        if self.exemplar_mode == "gold_rationale":
            return "ICL-EDR + gold rationale"
        return "ICL-EDR"

    def task_key(self, *parts: Any) -> str:
        return "::".join(safe_key(p) for p in parts)

    def temperature_for(self, setting: ModelSetting, method: str) -> float:
        if method == "cot" and setting.cot_temperature is not None:
            return float(setting.cot_temperature)
        if method == "icl" and setting.icl_temperature is not None:
            return float(setting.icl_temperature)
        return float(setting.temperature)

    def cot_task(self, row: pd.Series, setting: ModelSetting, rollout_idx: int) -> dict[str, Any]:
        return {
            "schema_version": "icl_edr_cot_v1",
            "run_name": self.run_name,
            "stage": "cot",
            "method": "CoT",
            "setting": setting.name,
            "provider": setting.provider,
            "hf_provider": setting.hf_provider,
            "model": setting.model_id,
            "temperature": self.temperature_for(setting, "cot"),
            "reasoning_effort": setting.reasoning_effort,
            "top_p": setting.top_p,
            "extra_body": setting.extra_body or {},
            "use_response_format": setting.use_response_format,
            "max_tokens": setting.max_tokens,
            "rollout_idx": rollout_idx,
            "task_key": self.task_key("cot", setting.name, row["dataset_name"], row["question_key"], rollout_idx),
            **self.question_metadata(row),
            "messages": [
                {"role": "system", "content": base_system_prompt(setting)},
                {"role": "user", "content": cot_user_prompt(row, setting, rollout_idx)},
            ],
            "response_schema": ANSWER_SCHEMA,
        }

    def solver_task(self, row: pd.Series, setting: ModelSetting, ensemble_repeat_idx: int) -> dict[str, Any]:
        exemplars = exemplars_for(row, self.retrieval, self.retrieval_k)
        if self.exemplar_mode == "gold_rationale":
            exemplars = attach_gold_rationales(exemplars, self.source_exemplar_cache)
        messages: list[dict[str, str]] = []
        control_prompt = setting_control_system_prompt(setting)
        if control_prompt:
            messages.append({"role": "system", "content": control_prompt})
        messages.append({"role": "user", "content": icl_solver_user_prompt(row, exemplars, setting, 0, ensemble_repeat_idx, self.exemplar_mode)})
        solver_indices = list(range(self.icl_solver_count))
        solver_task_keys = [
            self.task_key("icl_solver", setting.name, row["dataset_name"], row["question_key"], ensemble_repeat_idx, solver_idx)
            for solver_idx in solver_indices
        ]
        return {
            "schema_version": "icl_edr_solver_batch_v1",
            "run_name": self.run_name,
            "stage": "icl_solver_batch",
            "method": self.icl_method_name(),
            "setting": setting.name,
            "provider": setting.provider,
            "hf_provider": setting.hf_provider,
            "model": setting.model_id,
            "temperature": self.temperature_for(setting, "icl"),
            "reasoning_effort": setting.reasoning_effort,
            "top_p": setting.top_p,
            "extra_body": setting.extra_body or {},
            "use_response_format": setting.use_response_format,
            "max_tokens": setting.max_tokens,
            "ensemble_repeat_idx": ensemble_repeat_idx,
            "solver_indices": solver_indices,
            "solver_task_keys": solver_task_keys,
            "n": self.icl_solver_count,
            "task_key": self.task_key("icl_solver_batch", setting.name, row["dataset_name"], row["question_key"], ensemble_repeat_idx),
            "exemplar_mode": self.exemplar_mode,
            "exemplar_count": len(exemplars),
            "exemplars": exemplars,
            **self.question_metadata(row),
            "messages": messages,
            "response_schema": SOLVER_SCHEMA,
        }

    def solver_single_task(self, row: pd.Series, setting: ModelSetting, ensemble_repeat_idx: int, solver_idx: int) -> dict[str, Any]:
        task = self.solver_task(row, setting, ensemble_repeat_idx)
        task.update(
            {
                "schema_version": "icl_edr_solver_v1",
                "stage": "icl_solver",
                "solver_idx": solver_idx,
                "n": 1,
                "task_key": self.task_key("icl_solver", setting.name, row["dataset_name"], row["question_key"], ensemble_repeat_idx, solver_idx),
            }
        )
        task.pop("solver_indices", None)
        task.pop("solver_task_keys", None)
        return task

    def judge_task(self, row: pd.Series, setting: ModelSetting, solver_rows: list[dict[str, Any]], ensemble_repeat_idx: int, judge_repeat_idx: int) -> dict[str, Any]:
        return {
            "schema_version": "icl_edr_judge_v1",
            "run_name": self.run_name,
            "stage": "icl_judge",
            "method": self.icl_method_name(),
            "setting": setting.name,
            "provider": setting.provider,
            "hf_provider": setting.hf_provider,
            "model": setting.model_id,
            "temperature": self.temperature_for(setting, "icl"),
            "reasoning_effort": setting.reasoning_effort,
            "top_p": setting.top_p,
            "extra_body": setting.extra_body or {},
            "use_response_format": setting.use_response_format,
            "max_tokens": setting.max_tokens,
            "ensemble_repeat_idx": ensemble_repeat_idx,
            "judge_repeat_idx": judge_repeat_idx,
            "task_key": self.task_key("icl_judge", setting.name, row["dataset_name"], row["question_key"], ensemble_repeat_idx, judge_repeat_idx),
            "solver_task_keys": [r.get("task_key") for r in solver_rows],
            "exemplar_mode": self.exemplar_mode,
            "candidate_answers": sorted({clean_letter(r.get("selected_answer"), row["valid_options"]) for r in solver_rows if clean_letter(r.get("selected_answer"), row["valid_options"])}),
            **self.question_metadata(row),
            "messages": [
                {"role": "system", "content": icl_judge_system_prompt(setting)},
                {"role": "user", "content": judge_user_prompt(row, solver_rows, setting, ensemble_repeat_idx, judge_repeat_idx)},
            ],
            "response_schema": JUDGE_SCHEMA,
        }

    def question_metadata(self, row: pd.Series | dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": row.get("sample_id"),
            "dataset_name": row["dataset_name"],
            "dataset_family": row.get("dataset_family"),
            "dataset_split": row.get("dataset_split"),
            "language": row.get("language"),
            "question_key": row["question_key"],
            "canonical_question_key": row.get("canonical_question_key"),
            "question_index": int(row["question_index"]),
            "meta_info": None if pd.isna(row.get("meta_info")) else row.get("meta_info"),
            "pna_year": None if pd.isna(row.get("pna_year")) else row.get("pna_year"),
            "pna_part": None if pd.isna(row.get("pna_part")) else row.get("pna_part"),
            "pna_question_number": None if pd.isna(row.get("pna_question_number")) else row.get("pna_question_number"),
            "difficulty": row.get("difficulty"),
            "ground_truth": row["ground_truth"],
            "ground_truth_text": row.get("ground_truth_text"),
            "question_with_options": row["question_with_options"],
            "valid_options": row.get("valid_options") or VALID_OPTIONS,
            "options": row.get("options") or {},
        }

    def build_cot_and_solver_tasks(self) -> list[dict[str, Any]]:
        # Stage 1 always contains ICL-EDR solver tasks.
        # If include_cot_baseline=True, it also contains matched CoT baseline
        # tasks created by cot_task().
        #
        # With batch_solver_choices=True, the S solver completions are requested
        # in one provider call when the provider supports n completions.
        tasks = []
        for _, row in self.target_df.iterrows():
            for setting in self.settings:
                if self.include_cot_baseline:
                    for rollout_idx in range(self.cot_rollouts):
                        tasks.append(self.cot_task(row, setting, rollout_idx))
                for ensemble_repeat_idx in range(self.icl_ensemble_repeats):
                    if self.batch_solver_choices:
                        tasks.append(self.solver_task(row, setting, ensemble_repeat_idx))
                    else:
                        for solver_idx in range(self.icl_solver_count):
                            tasks.append(self.solver_single_task(row, setting, ensemble_repeat_idx, solver_idx))
        return tasks

    def build_judge_tasks(self, existing: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        tasks = []
        for _, row in self.target_df.iterrows():
            for setting in self.settings:
                for ensemble_repeat_idx in range(self.icl_ensemble_repeats):
                    solver_rows = self.solver_rows_for(row, setting.name, ensemble_repeat_idx, existing)
                    if len(solver_rows) < self.icl_solver_count:
                        continue
                    answers = [clean_letter(r.get("selected_answer"), row["valid_options"]) for r in solver_rows]
                    if len({a for a in answers if a}) <= 1:
                        continue
                    for judge_repeat_idx in range(self.icl_judge_repeats):
                        tasks.append(self.judge_task(row, setting, solver_rows, ensemble_repeat_idx, judge_repeat_idx))
        return tasks

    def get_client(self, setting: ModelSetting):
        if setting.provider == "openai":
            if not get_openai_api_key(self.project_root):
                raise RuntimeError("OPENAI_API_KEY is not set. Set OPENAI_API_KEY for OpenAI provider calls.")
            attr = "client_openai"
            if not hasattr(self.thread_local, attr):
                from openai import OpenAI

                setattr(self.thread_local, attr, OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=self.request_timeout))
            return getattr(self.thread_local, attr)

        token = get_hf_token(self.project_root)
        if not token:
            raise RuntimeError("HF_TOKEN is not set. Set HF_TOKEN for Hugging Face Inference Providers.")
        attr = f"client_{safe_key(setting.hf_provider)}"
        if not hasattr(self.thread_local, attr):
            from huggingface_hub import InferenceClient

            setattr(self.thread_local, attr, InferenceClient(provider=setting.hf_provider, token=token, timeout=self.request_timeout))
        return getattr(self.thread_local, attr)

    def call_chat(self, task: dict[str, Any], *, use_response_format: bool, use_extra_body: bool) -> Any:
        setting = self.setting_by_name(task["setting"])
        client = self.get_client(setting)
        payload: dict[str, Any] = {
            "model": task["model"],
            "messages": task["messages"],
            "temperature": task["temperature"],
            "stream": False,
        }
        if int(task.get("n") or 1) > 1:
            payload["n"] = int(task["n"])
        if task.get("top_p") is not None:
            payload["top_p"] = task["top_p"]
        if use_response_format and task.get("use_response_format"):
            payload["response_format"] = {"type": "json_schema", "json_schema": task["response_schema"]}
        if use_extra_body and task.get("extra_body"):
            payload["extra_body"] = task["extra_body"]
        if setting.provider == "openai":
            payload["max_completion_tokens"] = task["max_tokens"]
            if task.get("reasoning_effort"):
                payload["reasoning_effort"] = task["reasoning_effort"]
            try:
                return client.chat.completions.create(**payload)
            except Exception as exc:
                text = repr(exc).lower()
                if "max_completion_tokens" in text and "max_tokens" in text:
                    fallback = dict(payload)
                    fallback.pop("max_completion_tokens", None)
                    fallback["max_tokens"] = task["max_tokens"]
                    return client.chat.completions.create(**fallback)
                if "temperature" in text and ("unsupported" in text or "not support" in text or "invalid" in text):
                    fallback = dict(payload)
                    fallback.pop("temperature", None)
                    return client.chat.completions.create(**fallback)
                raise

        payload["max_tokens"] = task["max_tokens"]
        return client.chat.completions.create(**payload)

    def token_shares_for_solver_batch(self, usage: dict[str, Any], n: int) -> list[dict[str, int | None]]:
        prompt = usage.get("prompt_tokens")
        completion_parts = split_int(usage.get("completion_tokens"), n)
        reasoning_parts = split_int(usage.get("reasoning_tokens"), n)
        total = usage.get("total_tokens")
        if prompt is not None and usage.get("completion_tokens") is not None:
            prompt_parts = [int(prompt)] + [0] * (n - 1)
            total_parts = [
                (prompt_parts[i] if prompt_parts[i] is not None else 0) + (completion_parts[i] if completion_parts[i] is not None else 0)
                for i in range(n)
            ]
        else:
            prompt_parts = [prompt] + ([0] * (n - 1) if prompt is not None else [None] * (n - 1))
            total_parts = split_int(total, n)
        return [
            {
                "prompt_tokens": prompt_parts[i],
                "completion_tokens": completion_parts[i],
                "reasoning_tokens": reasoning_parts[i],
                "total_tokens": total_parts[i],
            }
            for i in range(n)
        ]

    def result_from_choice(
        self,
        task: dict[str, Any],
        choice: Any,
        usage: dict[str, Any],
        started_at: str,
        started: float,
        attempt: int,
        request_variant: str,
        *,
        overrides: dict[str, Any] | None = None,
        token_overrides: dict[str, int | None] | None = None,
    ) -> dict[str, Any]:
        content = choice.message.content or ""
        thinking_content, content_without_thinking = strip_thinking_blocks(content)
        parsed = extract_json_obj(content)
        selected = parsed.get("selected_answer") or parsed.get("answer_idx")
        selected = clean_letter(selected, task["valid_options"])
        if selected is None:
            raise ValueError(f"missing_or_invalid_selected_answer: {parsed}")
        row = {
            **{k: v for k, v in task.items() if k != "messages"},
            "messages": task["messages"],
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "elapsed_sec": time.perf_counter() - started,
            "attempt": attempt,
            "request_variant": request_variant,
            "raw_response": content,
            "thinking_content": thinking_content,
            "content_without_thinking": content_without_thinking,
            "parsed_response": parsed,
            "reasoning": parsed.get("reasoning") or parsed.get("decisive_distinction") or "",
            "selected_answer": selected,
            "selected_answer_text": parsed.get("selected_answer_text") or task.get("options", {}).get(selected, ""),
            "valid_prediction": True,
            "is_correct": selected == task["ground_truth"],
            "usage": usage,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "error": None,
        }
        if overrides:
            row.update(overrides)
        if token_overrides:
            row["batch_usage"] = usage
            row["usage"] = {**usage, **token_overrides, "allocated_from_batch": True}
            row.update(token_overrides)
        return row

    def solver_batch_overrides(self, task: dict[str, Any], choice_index: int) -> dict[str, Any]:
        solver_idx = int(task["solver_indices"][choice_index])
        return {
            "schema_version": "icl_edr_solver_v1",
            "stage": "icl_solver",
            "solver_idx": solver_idx,
            "task_key": task["solver_task_keys"][choice_index],
            "batch_task_key": task["task_key"],
            "batch_choice_index": choice_index,
            "n_completions": int(task.get("n") or 1),
        }

    def error_result(self, task: dict[str, Any], started_at: str, started: float, last_error: str | None) -> dict[str, Any]:
        return {
            **{k: v for k, v in task.items() if k != "messages"},
            "messages": task["messages"],
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "elapsed_sec": time.perf_counter() - started,
            "attempt": self.max_retries,
            "raw_response": "",
            "parsed_response": {},
            "selected_answer": None,
            "valid_prediction": False,
            "is_correct": False,
            "usage": {},
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
            "error": last_error,
        }

    def error_results_for_task(self, task: dict[str, Any], started_at: str, started: float, last_error: str | None) -> dict[str, Any] | list[dict[str, Any]]:
        if task.get("stage") != "icl_solver_batch":
            return self.error_result(task, started_at, started, last_error)
        rows = []
        for choice_index in range(len(task.get("solver_indices") or [])):
            row = self.error_result(task, started_at, started, last_error)
            row.update(self.solver_batch_overrides(task, choice_index))
            rows.append(row)
        return rows

    def invoke_task(self, task: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        started_at = utc_now_iso()
        started = time.perf_counter()
        last_error = None
        variants = [
            (bool(task.get("use_response_format")), bool(task.get("extra_body")), "configured"),
            (False, bool(task.get("extra_body")), "no_response_format"),
            (bool(task.get("use_response_format")), False, "no_extra_body"),
            (False, False, "minimal"),
        ]
        seen = set()
        variants = [v for v in variants if not (v[:2] in seen or seen.add(v[:2]))]
        for attempt in range(1, self.max_retries + 1):
            for use_response_format, use_extra_body, request_variant in variants:
                try:
                    response = self.call_chat(task, use_response_format=use_response_format, use_extra_body=use_extra_body)
                    plain = to_plain(response)
                    usage = normalize_usage(getattr(response, "usage", None) or (plain.get("usage") if isinstance(plain, dict) else None))
                    choices = list(response.choices)
                    if task.get("stage") == "icl_solver_batch":
                        n = int(task.get("n") or 1)
                        if len(choices) < n:
                            raise ValueError(f"expected_{n}_choices_got_{len(choices)}")
                        token_shares = self.token_shares_for_solver_batch(usage, n)
                        return [
                            self.result_from_choice(
                                task,
                                choices[choice_index],
                                usage,
                                started_at,
                                started,
                                attempt,
                                request_variant,
                                overrides=self.solver_batch_overrides(task, choice_index),
                                token_overrides=token_shares[choice_index],
                            )
                            for choice_index in range(n)
                        ]
                    return self.result_from_choice(task, choices[0], usage, started_at, started, attempt, request_variant)
                except Exception as exc:
                    last_error = repr(exc)
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff_seconds * attempt)
        return self.error_results_for_task(task, started_at, started, last_error)

    def setting_by_name(self, name: str) -> ModelSetting:
        for setting in self.settings:
            if setting.name == name:
                return setting
        raise KeyError(name)

    def load_existing_by_task(self) -> dict[str, dict[str, Any]]:
        rows = load_jsonl(self.outputs_jsonl)
        best: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row.get("task_key")
            if not key:
                continue
            old = best.get(key)
            if old is None or (not self.is_success(old) and self.is_success(row)):
                best[key] = row
        return best

    def is_success(self, row: dict[str, Any] | None) -> bool:
        return bool(row and row.get("error") is None and row.get("valid_prediction") and clean_letter(row.get("selected_answer"), row.get("valid_options") or VALID_OPTIONS))

    def task_is_success(self, task: dict[str, Any], existing: dict[str, dict[str, Any]]) -> bool:
        if task.get("stage") == "icl_solver_batch":
            return all(self.is_success(existing.get(key)) for key in task.get("solver_task_keys", []))
        return self.is_success(existing.get(task["task_key"]))

    def missing_tasks(self, tasks: list[dict[str, Any]], existing: dict[str, dict[str, Any]], *, apply_cap: bool = True) -> list[dict[str, Any]]:
        missing = [task for task in tasks if self.force_rerun or not self.task_is_success(task, existing)]
        if apply_cap and self.max_tasks_this_run > 0:
            missing = missing[: self.max_tasks_this_run]
        return missing

    def is_quota_error(self, text: Any) -> bool:
        low = str(text or "").lower()
        return any(term in low for term in ["quota", "billing", "payment", "insufficient", "exceeded", "rate limit"])

    def run_task_batch(self, tasks: list[dict[str, Any]], existing: dict[str, dict[str, Any]], label: str) -> None:
        all_missing = self.missing_tasks(tasks, existing, apply_cap=False)
        missing = all_missing[: self.max_tasks_this_run] if self.max_tasks_this_run > 0 else all_missing
        print(
            f"{label}: existing successes={len(tasks) - len(all_missing)} "
            f"missing_total={len(all_missing)} selected_this_run={len(missing)}"
        )
        if not missing:
            return
        if not self.run_api_calls:
            print(f"Dry run only. Set run_api_calls=True, or pass --run-api in the CLI, to execute provider calls.")
            print("Would run:", Counter((t["stage"], t["setting"], t["dataset_name"]) for t in missing))
            return

        providers_needed = {self.setting_by_name(task["setting"]).provider for task in missing}
        if "huggingface_inference_providers" in providers_needed and not get_hf_token(self.project_root):
            raise RuntimeError("HF_TOKEN is not set.")
        if "openai" in providers_needed and not get_openai_api_key(self.project_root):
            raise RuntimeError("OPENAI_API_KEY is not set.")

        completed = 0
        quota_stop = False
        for start in range(0, len(missing), self.batch_task_count):
            batch = missing[start : start + self.batch_task_count]
            print(f"{label} batch {start // self.batch_task_count + 1}: {start + 1}-{start + len(batch)} / {len(missing)}")
            with futures.ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(batch)))) as ex:
                fut_to_task = {ex.submit(self.invoke_task, task): task for task in batch}
                for fut in tqdm(futures.as_completed(fut_to_task), total=len(fut_to_task), desc=label):
                    result_or_rows = fut.result()
                    result_rows = result_or_rows if isinstance(result_or_rows, list) else [result_or_rows]
                    for result in result_rows:
                        append_jsonl(self.outputs_jsonl, result)
                        existing[result["task_key"]] = result
                    completed += 1
                    if self.stop_on_quota_error and any(self.is_quota_error(result.get("error")) for result in result_rows):
                        quota_stop = True
                        print("Quota/billing-like error detected; stopping after current cancellation.")
                        for other in fut_to_task:
                            other.cancel()
                        break
            self.write_summaries(existing)
            if quota_stop:
                break
        print(f"{label}: completed this run={completed}")

    def solver_rows_for(self, row: pd.Series | dict[str, Any], setting_name: str, ensemble_repeat_idx: int, existing: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for solver_idx in range(self.icl_solver_count):
            key = self.task_key("icl_solver", setting_name, row["dataset_name"], row["question_key"], ensemble_repeat_idx, solver_idx)
            old = existing.get(key)
            if self.is_success(old):
                rows.append(old)
        return rows

    def judge_rows_for(self, row: pd.Series | dict[str, Any], setting_name: str, ensemble_repeat_idx: int, existing: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for judge_repeat_idx in range(self.icl_judge_repeats):
            key = self.task_key("icl_judge", setting_name, row["dataset_name"], row["question_key"], ensemble_repeat_idx, judge_repeat_idx)
            old = existing.get(key)
            if self.is_success(old):
                rows.append(old)
        return rows

    def summarize_cot_rollout(self, row: pd.Series, setting: ModelSetting, rollout_idx: int, existing: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        key = self.task_key("cot", setting.name, row["dataset_name"], row["question_key"], rollout_idx)
        old = existing.get(key)
        if not self.is_success(old):
            return None
        selected = clean_letter(old.get("selected_answer"), row["valid_options"])
        vote_counts = {selected: 1} if selected else {}
        return self.result_row(row, setting, "CoT", selected, selected == row["ground_truth"], [old], False, "cot_single", vote_counts, rollout_idx)

    def summarize_icl_question(self, row: pd.Series, setting: ModelSetting, ensemble_repeat_idx: int, existing: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        solver_rows = self.solver_rows_for(row, setting.name, ensemble_repeat_idx, existing)
        if len(solver_rows) < self.icl_solver_count:
            return None
        solver_answers = [clean_letter(r.get("selected_answer"), row["valid_options"]) for r in solver_rows]
        majority, _, vote_counts = majority_vote(solver_answers)
        unique_answers = {a for a in solver_answers if a}
        if len(unique_answers) <= 1:
            selected = majority
            final_rows = solver_rows
            judge_ran = False
            final_source = "solver_agreement"
        else:
            judge_rows = self.judge_rows_for(row, setting.name, ensemble_repeat_idx, existing)
            if len(judge_rows) < self.icl_judge_repeats:
                return None
            judge_answers = [clean_letter(r.get("selected_answer"), row["valid_options"]) for r in judge_rows]
            selected, _, judge_counts = majority_vote(judge_answers)
            vote_counts = {"solver": vote_counts, "judge": judge_counts}
            final_rows = solver_rows + judge_rows
            judge_ran = True
            final_source = "routed_judge"
        return self.result_row(row, setting, self.icl_method_name(), selected, selected == row["ground_truth"], final_rows, judge_ran, final_source, vote_counts, ensemble_repeat_idx)

    def result_row(
        self,
        row: pd.Series,
        setting: ModelSetting,
        method: str,
        selected: str | None,
        is_correct: bool,
        source_rows: list[dict[str, Any]],
        judge_ran: bool,
        final_source: str,
        vote_counts: dict[str, Any],
        ensemble_repeat_idx: int | None = None,
    ) -> dict[str, Any]:
        request_keys = [r.get("batch_task_key") or r.get("task_key") for r in source_rows]
        api_calls = len({key for key in request_keys if key})
        elapsed_by_request: dict[str, float] = {}
        for source_row, request_key in zip(source_rows, request_keys):
            if not request_key:
                continue
            elapsed_by_request[str(request_key)] = max(
                elapsed_by_request.get(str(request_key), 0.0),
                float(source_row.get("elapsed_sec") or 0),
            )
        return {
            "run_name": self.run_name,
            "setting": setting.name,
            "method": method,
            "model": setting.model_id,
            "provider": setting.provider,
            "hf_provider": setting.hf_provider,
            "reasoning_effort": setting.reasoning_effort,
            "configured_temperature": setting.temperature,
            "cot_temperature": setting.cot_temperature,
            "icl_temperature": setting.icl_temperature,
            "dataset_name": row["dataset_name"],
            "dataset_family": row["dataset_family"],
            "language": row["language"],
            "sample_id": row["sample_id"],
            "question_key": row["question_key"],
            "question_index": int(row["question_index"]),
            "difficulty": row["difficulty"],
            "ground_truth": row["ground_truth"],
            "selected_answer": selected,
            "is_correct": bool(is_correct),
            "ensemble_repeat_idx": ensemble_repeat_idx,
            "judge_ran": judge_ran,
            "final_source": final_source,
            "vote_counts": json.dumps(vote_counts, ensure_ascii=False, sort_keys=True),
            "api_calls": api_calls,
            "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in source_rows),
            "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in source_rows),
            "reasoning_tokens": sum(int(r.get("reasoning_tokens") or 0) for r in source_rows),
            "total_tokens": sum(int(r.get("total_tokens") or 0) for r in source_rows),
            "elapsed_sec": sum(elapsed_by_request.values()),
            "source_task_keys": json.dumps([r.get("task_key") for r in source_rows], ensure_ascii=False),
        }

    def build_per_question_results(self, existing: dict[str, dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for _, row in self.target_df.iterrows():
            for setting in self.settings:
                if self.include_cot_baseline:
                    for rollout_idx in range(self.cot_rollouts):
                        cot = self.summarize_cot_rollout(row, setting, rollout_idx, existing)
                        if cot:
                            rows.append(cot)
                for ensemble_repeat_idx in range(self.icl_ensemble_repeats):
                    icl = self.summarize_icl_question(row, setting, ensemble_repeat_idx, existing)
                    if icl:
                        rows.append(icl)
        return pd.DataFrame(rows)

    def write_summaries(self, existing: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
        existing = existing or self.load_existing_by_task()
        per_question = self.build_per_question_results(existing)
        if per_question.empty:
            per_question.to_csv(self.per_question_csv, index=False)
            summary = pd.DataFrame()
            summary.to_csv(self.summary_csv, index=False)
        else:
            per_question.to_csv(self.per_question_csv, index=False)
            summary = (
                per_question.groupby(["setting", "method", "dataset_name"], dropna=False)
                .agg(
                    n=("is_correct", "size"),
                    accuracy=("is_correct", "mean"),
                    mean_total_tokens=("total_tokens", "mean"),
                    mean_api_calls=("api_calls", "mean"),
                    judge_rate=("judge_ran", "mean"),
                )
                .reset_index()
            )
            combined = (
                per_question.groupby(["setting", "method"], dropna=False)
                .agg(
                    n=("is_correct", "size"),
                    accuracy=("is_correct", "mean"),
                    mean_total_tokens=("total_tokens", "mean"),
                    mean_api_calls=("api_calls", "mean"),
                    judge_rate=("judge_ran", "mean"),
                )
                .reset_index()
            )
            combined["dataset_name"] = "combined"
            summary = pd.concat([summary, combined], ignore_index=True, sort=False)
            summary["accuracy_pct"] = summary["accuracy"] * 100
            summary.to_csv(self.summary_csv, index=False)
        payload = {
            "run_name": self.run_name,
            "settings": [asdict(s) for s in self.settings],
            "run_api_calls": self.run_api_calls,
            "include_cot_baseline": self.include_cot_baseline,
            "exemplar_mode": self.exemplar_mode,
            "source_exemplar_cache": None if self.source_exemplar_cache_path is None else str(self.source_exemplar_cache_path.resolve()),
            "target_n": int(len(self.target_df)),
            "output_rows": int(len(existing)),
            "completed_per_question_rows": int(len(per_question)),
            "outputs_jsonl": str(self.outputs_jsonl.resolve()),
            "per_question_csv": str(self.per_question_csv.resolve()),
            "summary_csv": str(self.summary_csv.resolve()),
            "updated_at_utc": utc_now_iso(),
        }
        self.summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return per_question

    def run(self) -> None:
        print("Project root:", self.project_root)
        print("Output dir:", self.out_dir)
        print("Settings:", [s.name for s in self.settings])
        print("Target rows:", len(self.target_df), self.target_df["dataset_name"].value_counts().to_dict())
        print("Run API calls:", self.run_api_calls)
        print("Include matched CoT baseline:", self.include_cot_baseline)
        # Save the exact prompt/schema/config snapshot beside the raw outputs.
        self.write_prompt_snapshot()

        # Run ICL-EDR solvers first; optionally include the matched CoT baseline.
        # Existing successful rows in outputs.jsonl are reused unless force_rerun=True.
        existing = self.load_existing_by_task()
        cot_solver_tasks = self.build_cot_and_solver_tasks()
        task_label = "CoT and ICL-EDR solver tasks" if self.include_cot_baseline else "ICL-EDR solver tasks"
        self.run_task_batch(cot_solver_tasks, existing, task_label)

        # Run the ICL-EDR judge only for questions where the solver ensemble
        # disagrees. If all solver completions agree, no judge call is made.
        existing = self.load_existing_by_task()
        judge_tasks = self.build_judge_tasks(existing)
        self.run_task_batch(judge_tasks, existing, "ICL-EDR routed judge tasks")

        # Convert raw model outputs into per-question rows and summary tables.
        existing = self.load_existing_by_task()
        per_question = self.write_summaries(existing)
        print("Per-question completed rows:", len(per_question))
        if not per_question.empty:
            summary = pd.read_csv(self.summary_csv)
            print(summary.to_string(index=False))


def gptoss_120b_deepinfra_settings() -> list[ModelSetting]:
    model_id = os.getenv("GPTOSS120B_DEEPINFRA_MODEL_ID", "openai/gpt-oss-120b")
    hf_provider = os.getenv("GPTOSS120B_DEEPINFRA_PROVIDER", "together")
    max_tokens = {
        "low": int(os.getenv("GPTOSS120B_LOW_MAX_TOKENS", "4096")),
        "medium": int(os.getenv("GPTOSS120B_MEDIUM_MAX_TOKENS", "8192")),
        "high": int(os.getenv("GPTOSS120B_HIGH_MAX_TOKENS", "12000")),
    }
    temperature = float(os.getenv("GPTOSS120B_TEMPERATURE", "1.0"))
    use_reasoning_effort_param = os.getenv("GPTOSS120B_USE_REASONING_EFFORT_PARAM", "0") == "1"
    return [
        ModelSetting(
            name=f"gpt_oss_120b_{level}",
            provider="huggingface_inference_providers",
            model_id=model_id,
            hf_provider=hf_provider,
            temperature=temperature,
            max_tokens=max_tokens[level],
            reasoning_effort=level if use_reasoning_effort_param else None,
            extra_body={"reasoning_effort": level} if use_reasoning_effort_param else None,
            use_response_format=os.getenv("GPTOSS120B_RESPONSE_FORMAT", "1") == "1",
            system_prefix=f"Reasoning: {level}",
            notes=(
                f"gpt-oss reasoning level {level} requested in system prompt"
                + (" and reasoning_effort payload parameter." if use_reasoning_effort_param else ".")
            ),
        )
        for level in ["low", "medium", "high"]
    ]


def qwen3_235b_nscale_settings() -> list[ModelSetting]:
    model_id = os.getenv("QWEN3_235B_NSCALE_MODEL_ID", "Qwen/Qwen3-235B-A22B")
    hf_provider = os.getenv("QWEN3_235B_NSCALE_PROVIDER", "nscale")
    return [
        ModelSetting(
            name="qwen3_235b_no_think",
            provider="huggingface_inference_providers",
            model_id=model_id,
            hf_provider=hf_provider,
            temperature=float(os.getenv("QWEN3_235B_NO_THINK_TEMPERATURE", "1")),
            cot_temperature=float(os.getenv("QWEN3_235B_NO_THINK_COT_TEMPERATURE", "0")),
            icl_temperature=float(os.getenv("QWEN3_235B_NO_THINK_ICL_TEMPERATURE", "1")),
            max_tokens=int(os.getenv("QWEN3_235B_NO_THINK_MAX_TOKENS", "4096")),
            use_response_format=os.getenv("QWEN3_235B_NO_THINK_RESPONSE_FORMAT", "1") == "1",
            user_prefix="/no_think",
            notes="Qwen no-thinking mode requested with /no_think; CoT uses temperature 0 and ICL-EDR uses temperature 1 by default.",
        ),
        ModelSetting(
            name="qwen3_235b_think",
            provider="huggingface_inference_providers",
            model_id=model_id,
            hf_provider=hf_provider,
            temperature=float(os.getenv("QWEN3_235B_THINK_TEMPERATURE", "0.6")),
            top_p=float(os.getenv("QWEN3_235B_THINK_TOP_P", "0.95")),
            extra_body={
                "top_k": int(os.getenv("QWEN3_235B_THINK_TOP_K", "20")),
                "min_p": float(os.getenv("QWEN3_235B_THINK_MIN_P", "0")),
            },
            max_tokens=int(os.getenv("QWEN3_235B_THINK_MAX_TOKENS", "12000")),
            use_response_format=os.getenv("QWEN3_235B_THINK_RESPONSE_FORMAT", "0") == "1",
            user_prefix="/think",
            notes="Qwen thinking mode requested with /think.",
        ),
    ]


def gpt54mini_openai_settings() -> list[ModelSetting]:
    model_id = os.getenv("GPT54MINI_OPENAI_MODEL", "gpt-5.4-mini")
    response_format = os.getenv("GPT54MINI_OPENAI_RESPONSE_FORMAT", "1") == "1"
    common_temperature = float(os.getenv("GPT54MINI_OPENAI_TEMPERATURE", "1.0"))
    return [
        ModelSetting(
            name="gpt54mini_no_reasoning",
            provider="openai",
            model_id=model_id,
            temperature=common_temperature,
            cot_temperature=float(os.getenv("GPT54MINI_NO_REASONING_COT_TEMPERATURE", "0")),
            icl_temperature=float(os.getenv("GPT54MINI_NO_REASONING_ICL_TEMPERATURE", "1")),
            max_tokens=int(os.getenv("GPT54MINI_NO_REASONING_MAX_TOKENS", "8192")),
            reasoning_effort=os.getenv("GPT54MINI_NO_REASONING_EFFORT", "none"),
            use_response_format=response_format,
            notes="GPT-5.4-mini no-reasoning setting; CoT uses temperature 0 and ICL-EDR uses temperature 1 by default.",
        ),
        ModelSetting(
            name="gpt54mini_medium",
            provider="openai",
            model_id=model_id,
            temperature=common_temperature,
            cot_temperature=float(os.getenv("GPT54MINI_MEDIUM_COT_TEMPERATURE", "1")),
            icl_temperature=float(os.getenv("GPT54MINI_MEDIUM_ICL_TEMPERATURE", "1")),
            max_tokens=int(os.getenv("GPT54MINI_MEDIUM_MAX_TOKENS", "8192")),
            reasoning_effort=os.getenv("GPT54MINI_MEDIUM_EFFORT", "medium"),
            use_response_format=response_format,
            notes="GPT-5.4-mini medium reasoning setting; CoT and ICL-EDR use temperature 1 by default.",
        ),
        ModelSetting(
            name="gpt54mini_xhigh",
            provider="openai",
            model_id=model_id,
            temperature=common_temperature,
            cot_temperature=float(os.getenv("GPT54MINI_XHIGH_COT_TEMPERATURE", "1")),
            icl_temperature=float(os.getenv("GPT54MINI_XHIGH_ICL_TEMPERATURE", "1")),
            max_tokens=int(os.getenv("GPT54MINI_XHIGH_MAX_TOKENS", "20000")),
            reasoning_effort=os.getenv("GPT54MINI_XHIGH_EFFORT", "xhigh"),
            use_response_format=response_format,
            notes="GPT-5.4-mini maximum reasoning setting; CoT and ICL-EDR use temperature 1 by default.",
        ),
    ]


def deepseek_v32_novita_settings() -> list[ModelSetting]:
    model_id = os.getenv("DEEPSEEK_V32_NOVITA_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    hf_provider = os.getenv("DEEPSEEK_V32_NOVITA_PROVIDER", "novita")
    return [
        ModelSetting(
            name=f"deepseek_v32_{mode}",
            provider="huggingface_inference_providers",
            model_id=model_id,
            hf_provider=hf_provider,
            temperature=float(os.getenv("DEEPSEEK_V32_TEMPERATURE", "1.0")),
            top_p=float(os.getenv("DEEPSEEK_V32_TOP_P", "0.95")),
            max_tokens=int(os.getenv("DEEPSEEK_V32_MAX_TOKENS", "12000")),
            extra_body={"thinking_mode": mode},
            use_response_format=os.getenv("DEEPSEEK_V32_RESPONSE_FORMAT", "0") == "1",
            notes=(
                f"DeepSeek-V3.2 {mode} mode requested through provider extra_body; "
                "defaults follow the model-card local-deployment sampling recommendation."
            ),
        )
        for mode in ["chat", "thinking"]
    ]


def make_gptoss_runner() -> ICLEDRRunner:
    return ICLEDRRunner.from_env(
        run_name=os.getenv("GPTOSS120B_DEEPINFRA_RUN_NAME", "gptoss120b_together_cot_icl_edr_v1"),
        settings=gptoss_120b_deepinfra_settings(),
        env_prefix="GPTOSS120B_DEEPINFRA_ICL_EDR",
    )


def make_qwen3_235b_runner() -> ICLEDRRunner:
    return ICLEDRRunner.from_env(
        run_name=os.getenv("QWEN3_235B_NSCALE_RUN_NAME", "qwen3_235b_nscale_cot_icl_edr_v1"),
        settings=qwen3_235b_nscale_settings(),
        env_prefix="QWEN3_235B_NSCALE_ICL_EDR",
    )


def make_gpt54mini_openai_runner() -> ICLEDRRunner:
    return ICLEDRRunner.from_env(
        run_name=os.getenv("GPT54MINI_OPENAI_RUN_NAME", "gpt54mini_openai_cot_icl_edr_v1"),
        settings=gpt54mini_openai_settings(),
        env_prefix="GPT54MINI_OPENAI_ICL_EDR",
    )


def make_deepseek_v32_novita_runner() -> ICLEDRRunner:
    return ICLEDRRunner.from_env(
        run_name=os.getenv("DEEPSEEK_V32_NOVITA_RUN_NAME", "deepseek_v32_novita_chat_thinking_cot_icl_edr_v1"),
        settings=deepseek_v32_novita_settings(),
        env_prefix="DEEPSEEK_V32_NOVITA_ICL_EDR",
    )


__all__ = [
    "ICLEDRRunner",
    "ModelSetting",
    "make_gptoss_runner",
    "make_qwen3_235b_runner",
    "make_gpt54mini_openai_runner",
    "make_deepseek_v32_novita_runner",
    "gptoss_120b_deepinfra_settings",
    "qwen3_235b_nscale_settings",
    "gpt54mini_openai_settings",
    "deepseek_v32_novita_settings",
]
