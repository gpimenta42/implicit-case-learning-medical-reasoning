from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


VALID_OPTIONS = ["A", "B", "C", "D", "E"]


EmbeddingFn = Callable[[list[str]], np.ndarray]


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def first_present(row: pd.Series | dict[str, Any], names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in row and not is_missing(row[name]):
            return row[name]
    return default


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


def normalized_options(row: pd.Series | dict[str, Any]) -> dict[str, str]:
    options = parse_jsonish(row.get("options"), default=None) if "options" in row else None
    if isinstance(options, dict):
        return {str(k).strip().upper(): str(v).strip() for k, v in options.items() if str(k).strip().upper() in VALID_OPTIONS}

    option_cols = {
        letter: first_present(row, [f"option_{letter}", f"option_{letter.lower()}", letter], default=None)
        for letter in VALID_OPTIONS
    }
    option_cols = {letter: str(text).strip() for letter, text in option_cols.items() if not is_missing(text)}
    if option_cols:
        return option_cols

    _, parsed = parse_options_from_question(str(first_present(row, ["question_with_options"], default="")))
    return parsed


def normalized_question(row: pd.Series | dict[str, Any]) -> str:
    question = first_present(row, ["question_stem", "question"], default=None)
    if not is_missing(question):
        return str(question).strip()
    stem, _ = parse_options_from_question(str(first_present(row, ["question_with_options"], default="")))
    return stem


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))


def retrieval_text(row: pd.Series | dict[str, Any]) -> str:
    question = normalized_question(row)
    options = normalized_options(row)
    option_text = format_options(options)
    return f"{question}\n\nOptions:\n{option_text}".strip() if option_text else question


def target_key(row: pd.Series | dict[str, Any]) -> str:
    return str(first_present(row, ["canonical_question_key", "question_key", "sample_id"], default="")).strip()


def source_question_key(row: pd.Series | dict[str, Any]) -> str:
    return str(first_present(row, ["question_key", "canonical_question_key", "sample_id"], default="")).strip()


def source_case_id(row: pd.Series | dict[str, Any]) -> str:
    existing = first_present(row, ["source_case_id"], default=None)
    if not is_missing(existing):
        return str(existing).strip()

    qkey = source_question_key(row)
    dataset_family = str(first_present(row, ["dataset_family"], default="")).strip().lower()
    dataset_name = str(first_present(row, ["dataset_name", "source_dataset"], default="")).strip().lower()

    if qkey.startswith("pna:"):
        return qkey
    if dataset_family == "medqa" or "medqa" in dataset_name:
        return f"medqa:{qkey}" if qkey else ""
    if dataset_family == "pna" or "pna" in dataset_name:
        return f"pna:{qkey}" if qkey and not qkey.startswith("pna:") else qkey
    return qkey


def answer_letter(row: pd.Series | dict[str, Any]) -> str:
    value = first_present(row, ["correct_answer_idx", "gold_answer_idx", "ground_truth", "answer_idx"], default="")
    text = str(value).strip().upper()
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else text


def answer_text(row: pd.Series | dict[str, Any], options: dict[str, str]) -> str:
    direct = first_present(row, ["correct_answer_text", "gold_answer_text", "ground_truth_text", "answer_text"], default=None)
    if not is_missing(direct):
        return str(direct).strip()
    return options.get(answer_letter(row), "")


def source_dataset(row: pd.Series | dict[str, Any]) -> str:
    return str(first_present(row, ["source_dataset", "dataset_name", "dataset_family"], default="")).strip()


def source_index(row: pd.Series | dict[str, Any]) -> Any:
    return first_present(row, ["source_index", "question_index", "index"], default=None)


def json_ready(value: Any) -> Any:
    if is_missing(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


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


def load_env_from_cwd_and_home() -> None:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        load_env_file(candidate / ".env")
    load_env_file(Path.home() / ".env")


def embed_texts_openai(texts: list[str], *, model: str = "text-embedding-3-large", batch_size: int = 128) -> np.ndarray:
    from openai import OpenAI

    load_env_from_cwd_and_home()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it in the environment, repository .env, or ~/.env before building retrieval embeddings.")

    client = OpenAI(api_key=api_key)
    embeddings: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        embeddings.extend(item.embedding for item in ordered)
    return np.asarray(embeddings, dtype=np.float32)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def source_entry(
    row: pd.Series,
    *,
    rank: int,
    dense_similarity: float,
    embedding_model: str,
    source_pool_mode: str,
) -> dict[str, Any]:
    options = normalized_options(row)
    return {
        "source_case_id": source_case_id(row),
        "source_dataset": source_dataset(row),
        "source_pool_mode": source_pool_mode,
        "source_index": json_ready(source_index(row)),
        "question_key": source_question_key(row),
        "dataset_split": json_ready(first_present(row, ["dataset_split"], default=None)),
        "question": normalized_question(row),
        "options": options,
        "correct_answer_idx": answer_letter(row),
        "correct_answer_text": answer_text(row, options),
        "rank": int(rank),
        "dense_similarity": float(dense_similarity),
        "embedding_model": embedding_model,
        "meta_info": json_ready(first_present(row, ["meta_info"], default=None)),
        "pna_year": json_ready(first_present(row, ["pna_year"], default=None)),
        "pna_part": json_ready(first_present(row, ["pna_part"], default=None)),
        "pna_question_number": json_ready(first_present(row, ["pna_question_number"], default=None)),
    }


def build_retrieval_json(
    *,
    target_csv_path: str | Path,
    source_csv_path: str | Path,
    output_json_path: str | Path | None = None,
    top_k: int = 5,
    embedding_model: str = "text-embedding-3-large",
    source_pool_mode: str = "custom_source_pool",
    embedding_fn: EmbeddingFn | None = None,
    openai_batch_size: int = 128,
    exclude_same_question: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Build the ICL-EDR retrieval JSON consumed by ``ICLEDRRunner``.

    ``target_csv_path`` should contain the target evaluation questions. ``source_csv_path``
    should contain the labelled source cases available for retrieval. Both files can
    use the same broad schema as the target-question CSV: question text/options, a
    question key, gold answer letter, and gold answer text.

    By default this function uses OpenAI embeddings. For tests or alternative
    embedding systems, pass ``embedding_fn``; it receives the concatenated list of
    target and source retrieval texts and must return a 2D numpy-compatible array.
    """
    target_df = pd.read_csv(target_csv_path)
    source_df = pd.read_csv(source_csv_path)
    if target_df.empty:
        raise ValueError("target_csv_path contains no rows.")
    if source_df.empty:
        raise ValueError("source_csv_path contains no rows.")

    target_texts = [retrieval_text(row) for _, row in target_df.iterrows()]
    source_texts = [retrieval_text(row) for _, row in source_df.iterrows()]
    all_texts = target_texts + source_texts

    if embedding_fn is None:
        embeddings = embed_texts_openai(all_texts, model=embedding_model, batch_size=openai_batch_size)
    else:
        embeddings = np.asarray(embedding_fn(all_texts), dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(all_texts):
        raise ValueError(f"Expected embeddings with shape ({len(all_texts)}, dim), got {embeddings.shape}.")

    target_emb = l2_normalize(embeddings[: len(target_texts)])
    source_emb = l2_normalize(embeddings[len(target_texts) :])
    similarity = target_emb @ source_emb.T

    output: dict[str, list[dict[str, Any]]] = {}
    source_keys = [source_question_key(row) for _, row in source_df.iterrows()]
    source_ids = [source_case_id(row) for _, row in source_df.iterrows()]

    for target_idx, (_, target_row) in enumerate(target_df.iterrows()):
        key = target_key(target_row)
        if not key:
            raise ValueError(f"Target row {target_idx} has no question key.")

        same_keys = {target_key(target_row), source_question_key(target_row), source_case_id(target_row)}
        ranked_source_indices = np.argsort(-similarity[target_idx])
        selected: list[dict[str, Any]] = []
        for source_idx in ranked_source_indices:
            if exclude_same_question and (source_keys[source_idx] in same_keys or source_ids[source_idx] in same_keys):
                continue
            source_row = source_df.iloc[int(source_idx)]
            selected.append(
                source_entry(
                    source_row,
                    rank=len(selected) + 1,
                    dense_similarity=float(similarity[target_idx, source_idx]),
                    embedding_model=embedding_model,
                    source_pool_mode=source_pool_mode,
                )
            )
            if len(selected) >= top_k:
                break
        output[key] = selected

    if output_json_path is not None:
        path = Path(output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return output


__all__ = ["build_retrieval_json", "embed_texts_openai"]
