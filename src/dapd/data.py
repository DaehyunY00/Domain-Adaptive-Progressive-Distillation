from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase

TOKENIZATION_PREPROCESSING_VERSION = "v1"


PROMPT_TEMPLATE = """### Domain Task
{prompt}

### Response
"""


@dataclass
class PreparedDatasets:
    train_lm: Dataset
    validation_lm: Dataset
    test_lm: Dataset
    validation_text: Dataset
    test_text: Dataset


class CausalLMDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [f["labels"] for f in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]

        padded_labels = []
        for label in labels:
            if len(label) < max_len:
                label = label + [-100] * (max_len - len(label))
            padded_labels.append(label)

        batch["labels"] = batch["input_ids"].new_tensor(padded_labels)
        return batch


def prepare_datasets(config: Any, tokenizer: PreTrainedTokenizerBase) -> PreparedDatasets:
    unified = build_unified_dataset(config)
    return prepare_datasets_from_unified(unified, config=config, tokenizer=tokenizer)


def prepare_datasets_from_unified(
    unified: DatasetDict,
    config: Any,
    tokenizer: PreTrainedTokenizerBase,
) -> PreparedDatasets:
    dataset_names = list(getattr(config, "datasets", []))
    seed = int(getattr(config, "seed", 42))
    preprocessing_version = str(
        getattr(config, "preprocessing_version", TOKENIZATION_PREPROCESSING_VERSION)
    )

    train_lm = tokenize_for_causal_lm(
        unified["train"],
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_proc=config.num_proc,
        split_name="train",
        dataset_names=dataset_names,
        seed=seed,
        preprocessing_version=preprocessing_version,
        tokenized_cache_dir=config.tokenized_cache_dir,
        enable_map_cache=config.enable_map_cache,
    )
    validation_lm = tokenize_for_causal_lm(
        unified["validation"],
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_proc=config.num_proc,
        split_name="validation",
        dataset_names=dataset_names,
        seed=seed,
        preprocessing_version=preprocessing_version,
        tokenized_cache_dir=config.tokenized_cache_dir,
        enable_map_cache=config.enable_map_cache,
    )
    test_lm = tokenize_for_causal_lm(
        unified["test"],
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_proc=config.num_proc,
        split_name="test",
        dataset_names=dataset_names,
        seed=seed,
        preprocessing_version=preprocessing_version,
        tokenized_cache_dir=config.tokenized_cache_dir,
        enable_map_cache=config.enable_map_cache,
    )

    return PreparedDatasets(
        train_lm=train_lm,
        validation_lm=validation_lm,
        test_lm=test_lm,
        validation_text=unified["validation"],
        test_text=unified["test"],
    )


def build_unified_dataset(config: Any) -> DatasetDict:
    split_collector: dict[str, list[Dataset]] = {"train": [], "validation": [], "test": []}

    for dataset_name in config.datasets:
        single = _load_domain_dataset(dataset_name, config.cache_dir)
        for split in split_collector:
            if split in single:
                split_collector[split].append(single[split])

    if not split_collector["train"]:
        raise ValueError("No train split found. Check dataset names in config.data.datasets")

    merged: dict[str, Dataset] = {}
    for split, datasets_list in split_collector.items():
        if datasets_list:
            merged[split] = concatenate_datasets(datasets_list).shuffle(seed=config.seed)

    if "validation" not in merged:
        split_data = merged["train"].train_test_split(test_size=0.1, seed=config.seed)
        merged["train"] = split_data["train"]
        merged["validation"] = split_data["test"]

    if "test" not in merged:
        split_data = merged["train"].train_test_split(test_size=0.1, seed=config.seed + 1)
        merged["train"] = split_data["train"]
        merged["test"] = split_data["test"]

    if config.max_train_samples:
        merged["train"] = merged["train"].select(range(min(config.max_train_samples, len(merged["train"]))))

    if config.max_eval_samples:
        merged["validation"] = merged["validation"].select(
            range(min(config.max_eval_samples, len(merged["validation"])))
        )
        merged["test"] = merged["test"].select(range(min(config.max_eval_samples, len(merged["test"]))))

    return DatasetDict(merged)


def build_external_eval_dataset(
    dataset_name: str,
    cache_dir: str | None,
    seed: int = 42,
    max_eval_samples: int | None = None,
) -> Dataset:
    """Load one dataset for external/OOD evaluation and return a single test dataset."""
    loaded = _load_domain_dataset(dataset_name, cache_dir)

    if "test" in loaded:
        test_ds = loaded["test"]
    elif "validation" in loaded:
        test_ds = loaded["validation"]
    elif "train" in loaded:
        test_ds = loaded["train"]
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' does not expose train/validation/test splits for OOD evaluation."
        )

    test_ds = test_ds.shuffle(seed=int(seed))
    if max_eval_samples is not None:
        test_ds = test_ds.select(range(min(int(max_eval_samples), len(test_ds))))
    return test_ds


def tokenize_for_causal_lm(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    num_proc: int,
    split_name: str,
    tokenized_cache_dir: str = "runs/dapd/cache/tokenized",
    enable_map_cache: bool = True,
    dataset_names: list[str] | tuple[str, ...] | None = None,
    seed: int = 42,
    preprocessing_version: str = TOKENIZATION_PREPROCESSING_VERSION,
) -> Dataset:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize(example: dict[str, str]) -> dict[str, Any]:
        prompt_text = PROMPT_TEMPLATE.format(prompt=example["prompt"])
        target = str(example["target"]).strip()
        full_text = f"{prompt_text}{target}"

        full_enc = tokenizer(full_text, truncation=True, max_length=max_length)
        prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)

        labels = list(full_enc["input_ids"])
        mask_until = min(len(prompt_enc["input_ids"]), len(labels))
        for i in range(mask_until):
            labels[i] = -100

        return {
            "input_ids": full_enc["input_ids"],
            "attention_mask": full_enc["attention_mask"],
            "labels": labels,
        }

    cache_file = _tokenize_cache_file(
        tokenized_cache_dir=tokenized_cache_dir,
        tokenizer=tokenizer,
        dataset=dataset,
        dataset_names=dataset_names,
        seed=seed,
        max_length=max_length,
        split_name=split_name,
        preprocessing_version=preprocessing_version,
    )
    if enable_map_cache and cache_file.exists():
        return Dataset.from_file(str(cache_file))

    map_num_proc = int(num_proc) if int(num_proc) > 1 else None
    tokenized = dataset.map(
        _tokenize,
        remove_columns=dataset.column_names,
        num_proc=map_num_proc,
        load_from_cache_file=enable_map_cache,
        cache_file_name=str(cache_file),
    )
    return tokenized


def _load_domain_dataset(dataset_name: str, cache_dir: str | None) -> DatasetDict:
    key = dataset_name.lower().strip()

    if key == "pubmedqa" or key == "pubmed_qa":
        raw = load_dataset("pubmed_qa", "pqa_labeled", cache_dir=cache_dir)
        mapper = _map_pubmedqa
    elif key == "sciq":
        raw = load_dataset("sciq", cache_dir=cache_dir)
        mapper = _map_sciq
    elif key == "medmcqa":
        raw = load_dataset("medmcqa", cache_dir=cache_dir)
        mapper = _map_medmcqa
    elif key == "bioasq":
        raw = _load_bioasq(cache_dir=cache_dir)
        mapper = _map_bioasq
    else:
        raise ValueError(
            "Unsupported dataset "
            f"'{dataset_name}'. Supported: pubmed_qa, sciq, medmcqa, bioasq"
        )

    out: dict[str, Dataset] = {}
    for split in raw.keys():
        ds = raw[split]
        mapped = ds.map(
            lambda x: mapper(x),
            remove_columns=ds.column_names,
        )
        mapped = mapped.filter(lambda x: bool(x["prompt"]) and bool(x["target"]))
        out[split] = mapped

    return DatasetDict(out)


def _map_pubmedqa(ex: dict[str, Any]) -> dict[str, str]:
    question = str(ex.get("question", "")).strip()
    contexts = ex.get("contexts", []) or ex.get("context", [])
    if isinstance(contexts, dict):
        context_text = "\n".join(str(v) for v in contexts.values() if v)
    elif isinstance(contexts, list):
        context_text = "\n".join(str(v) for v in contexts if v)
    else:
        context_text = str(contexts)

    decision = ex.get("final_decision", ex.get("answer", ""))
    long_answer = ex.get("long_answer", "")

    target = str(long_answer).strip() if long_answer else str(decision).strip()
    prompt = f"[Biomedical QA]\nQuestion: {question}\n\nContext:\n{context_text}\n\nAnswer:"

    return {"domain": "pubmed_qa", "prompt": prompt, "target": target}


def _map_sciq(ex: dict[str, Any]) -> dict[str, str]:
    support = str(ex.get("support", "")).strip()
    question = str(ex.get("question", "")).strip()
    answer = str(ex.get("correct_answer", "")).strip()

    prompt = f"[Scientific Reasoning]\nQuestion: {question}\n\nReference:\n{support}\n\nAnswer:"
    return {"domain": "sciq", "prompt": prompt, "target": answer}


def _map_medmcqa(ex: dict[str, Any]) -> dict[str, str]:
    question = str(ex.get("question", "")).strip()
    options = [
        str(ex.get("opa", "")).strip(),
        str(ex.get("opb", "")).strip(),
        str(ex.get("opc", "")).strip(),
        str(ex.get("opd", "")).strip(),
    ]

    cop = ex.get("cop", None)
    answer = ""
    if isinstance(cop, int) and 0 <= cop < len(options):
        answer = options[cop]
    elif isinstance(cop, int) and 1 <= cop <= len(options):
        answer = options[cop - 1]
    elif isinstance(cop, str) and cop.isdigit() and int(cop) < len(options):
        answer = options[int(cop)]
    elif isinstance(cop, str) and cop.isdigit() and 1 <= int(cop) <= len(options):
        answer = options[int(cop) - 1]
    else:
        answer = str(ex.get("exp", "")).strip()

    choices = "\n".join([f"{idx}. {opt}" for idx, opt in enumerate(options)])
    prompt = f"[Medical QA]\nQuestion: {question}\n\nChoices:\n{choices}\n\nAnswer:"
    return {"domain": "medmcqa", "prompt": prompt, "target": answer}


def _load_bioasq(cache_dir: str | None) -> DatasetDict:
    """Load BioASQ from HF Hub using a permissive config fallback chain."""
    candidates: list[tuple[str, str | None]] = [
        ("bioasq", None),
        ("bioasq", "Task10BGoldenEnriched"),
        ("bioasq", "Task10B"),
    ]

    last_error: Exception | None = None
    for name, config_name in candidates:
        try:
            if config_name is None:
                raw = load_dataset(name, cache_dir=cache_dir)
            else:
                raw = load_dataset(name, config_name, cache_dir=cache_dir)
            if not isinstance(raw, DatasetDict):
                continue
            return raw
        except Exception as exc:  # pragma: no cover - depends on remote dataset metadata.
            last_error = exc
            continue

    msg = "Failed to load BioASQ dataset from Hugging Face (`bioasq`)."
    if last_error is not None:
        msg += f" Last error: {last_error}"
    raise ValueError(msg)


def _map_bioasq(ex: dict[str, Any]) -> dict[str, str]:
    question = str(ex.get("question", ex.get("body", ""))).strip()

    snippets = ex.get("snippets", [])
    context_parts: list[str] = []
    if isinstance(snippets, list):
        for row in snippets:
            if isinstance(row, dict):
                txt = str(row.get("text", "")).strip()
            else:
                txt = str(row).strip()
            if txt:
                context_parts.append(txt)

    context_text = "\n".join(context_parts)

    ideal = ex.get("ideal_answer", "")
    if isinstance(ideal, list):
        ideal = " ".join(str(x) for x in ideal if x)
    ideal = str(ideal).strip()

    exact = ex.get("exact_answer", "")
    if isinstance(exact, list):
        exact = " ".join(str(x) for x in exact if x)
    exact = str(exact).strip()

    target = ideal or exact
    prompt = f"[BioASQ OOD QA]\nQuestion: {question}\n\nContext:\n{context_text}\n\nAnswer:"
    return {"domain": "bioasq", "prompt": prompt, "target": target}


def _tokenize_cache_file(
    tokenized_cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    dataset_names: list[str] | tuple[str, ...] | None,
    seed: int,
    max_length: int,
    split_name: str,
    preprocessing_version: str,
) -> Path:
    out_dir = Path(tokenized_cache_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_key = ",".join(str(name).strip().lower() for name in (dataset_names or []))
    dataset_fingerprint = str(getattr(dataset, "_fingerprint", "no_fingerprint"))
    tokenizer_id = _tokenizer_cache_id(tokenizer)
    template_hash = hashlib.sha1(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:8]
    key = (
        f"v={preprocessing_version}|datasets={dataset_key}|seed={seed}|split={split_name}|"
        f"max_len={max_length}|tokenizer={tokenizer_id}|prompt={template_hash}|"
        f"data_fp={dataset_fingerprint}"
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return out_dir / f"{split_name}_{digest}.arrow"


def _tokenizer_cache_id(tokenizer: PreTrainedTokenizerBase) -> str:
    name = str(getattr(tokenizer, "name_or_path", "tokenizer"))
    vocab_size = str(getattr(tokenizer, "vocab_size", "na"))
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    revision = str(init_kwargs.get("revision", "default"))
    tok_class = tokenizer.__class__.__name__
    return f"{tok_class}:{name}@{revision}:vocab={vocab_size}"
