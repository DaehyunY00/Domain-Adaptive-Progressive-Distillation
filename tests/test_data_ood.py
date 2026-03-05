from __future__ import annotations

from types import SimpleNamespace

from datasets import Dataset, DatasetDict

from dapd.data import (
    _load_domain_dataset,
    _map_bioasq,
    _map_medmcqa,
    build_external_eval_dataset,
    build_ood_dataset,
)


def test_build_external_eval_dataset_prefers_test_split(monkeypatch) -> None:
    train = Dataset.from_dict({"domain": ["x"], "prompt": ["p-train"], "target": ["a-train"]})
    test = Dataset.from_dict(
        {
            "domain": ["x", "x"],
            "prompt": ["p-test-1", "p-test-2"],
            "target": ["a-test-1", "a-test-2"],
        }
    )

    monkeypatch.setattr(
        "dapd.data._load_domain_dataset",
        lambda *args, **kwargs: DatasetDict({"train": train, "test": test}),
    )

    out = build_external_eval_dataset(
        dataset_name="bioasq",
        cache_dir=None,
        seed=42,
        max_eval_samples=1,
    )

    assert len(out) == 1
    assert set(out.column_names) == {"domain", "prompt", "target"}


def test_build_ood_dataset_merges_and_caps(monkeypatch) -> None:
    ds_a = Dataset.from_dict({"domain": ["a"], "prompt": ["p-a"], "target": ["t-a"]})
    ds_b = Dataset.from_dict(
        {
            "domain": ["b", "b"],
            "prompt": ["p-b1", "p-b2"],
            "target": ["t-b1", "t-b2"],
        }
    )

    def _fake_load(name: str, _cache_dir: str | None) -> DatasetDict:
        if name == "bioasq":
            return DatasetDict({"test": ds_a})
        if name == "pubmed_qa":
            return DatasetDict({"test": ds_b})
        raise ValueError(name)

    monkeypatch.setattr("dapd.data._load_domain_dataset", _fake_load)
    cfg = SimpleNamespace(ood_datasets=["bioasq", "pubmed_qa"], cache_dir=None, seed=42, ood_max_samples=2)

    out = build_ood_dataset(cfg)
    assert out is not None
    assert "test" in out
    assert len(out["test"]) == 2


def test_map_bioasq_prefers_exact_answer() -> None:
    ex = {
        "body": "What is BRCA1?",
        "exact_answer": ["Tumor suppressor gene"],
        "ideal_answer": ["Long form answer"],
    }
    mapped = _map_bioasq(ex)
    assert mapped["domain"] == "bioasq"
    assert "What is BRCA1?" in mapped["prompt"]
    assert mapped["target"] == "Tumor suppressor gene"


def test_load_domain_dataset_bioasq_fallback_to_pubmedqa(monkeypatch) -> None:
    pubmed = Dataset.from_dict(
        {
            "question": ["q1"],
            "contexts": [["c1"]],
            "final_decision": ["yes"],
            "long_answer": [""],
        }
    )

    def _fake_load_dataset(name: str, *args, **kwargs):
        del args, kwargs
        if name == "kroshan/BioASQ":
            raise RuntimeError("missing dataset")
        if name == "pubmed_qa":
            return DatasetDict({"train": pubmed})
        raise ValueError(name)

    monkeypatch.setattr("dapd.data.load_dataset", _fake_load_dataset)
    out = _load_domain_dataset("bioasq", cache_dir=None)
    assert "train" in out
    assert len(out["train"]) == 1
    assert out["train"][0]["domain"] == "pubmed_qa"


def test_map_medmcqa_prefers_one_indexed_cop() -> None:
    ex = {
        "question": "Q",
        "opa": "A",
        "opb": "B",
        "opc": "C",
        "opd": "D",
        "cop": 1,
        "exp": "fallback",
    }
    mapped = _map_medmcqa(ex)
    assert mapped["target"] == "A"
