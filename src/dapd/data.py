from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase


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
    train_lm = tokenize_for_causal_lm(
        unified["train"],
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_proc=config.num_proc,
        split_name="train",
        tokenized_cache_dir=config.tokenized_cache_dir,
        enable_map_cache=config.enable_map_cache,
    )
    validation_lm = tokenize_for_causal_lm(
        unified["validation"],
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_proc=config.num_proc,
        split_name="validation",
        tokenized_cache_dir=config.tokenized_cache_dir,
        enable_map_cache=config.enable_map_cache,
    )
    test_lm = tokenize_for_causal_lm(
        unified["test"],
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_proc=config.num_proc,
        split_name="test",
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


def tokenize_for_causal_lm(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    num_proc: int,
    split_name: str,
    tokenized_cache_dir: str,
    enable_map_cache: bool,
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
        max_length=max_length,
        split_name=split_name,
    )
    tokenized = dataset.map(
        _tokenize,
        remove_columns=dataset.column_names,
        num_proc=max(1, num_proc),
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
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported: pubmed_qa, sciq, medmcqa"
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


def _tokenize_cache_file(
    tokenized_cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    split_name: str,
) -> Path:
    out_dir = Path(tokenized_cache_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_name = getattr(tokenizer, "name_or_path", "tokenizer")
    key = f"{tok_name}|{tokenizer.vocab_size}|{max_length}|{split_name}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return out_dir / f"{split_name}_{digest}.arrow"
