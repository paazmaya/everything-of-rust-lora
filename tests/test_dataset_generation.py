"""Reproducible, source-grouped validation splitting."""

import copy
import importlib
import json

import pytest

generator = importlib.import_module("scripts.08_create_dataset")


def sample(group, output, category="general", **extra):
    return {
        "instruction": f"Explain {group}",
        "input": "",
        "output": output,
        "_group": group,
        "_category": category,
        **extra,
    }


def test_source_group_uses_document_version_not_fragment():
    chunk = {
        "source_type": "official_docs",
        "library": "rust",
        "version": "1.80",
        "url": "https://example.org/doc#one",
    }
    assert generator.source_group(chunk) == generator.source_group(
        {**chunk, "url": "https://example.org/doc#two"}
    )
    assert generator.source_group(chunk) != generator.source_group({**chunk, "version": "1.81"})
    assert generator.source_group({}) == generator.source_group({})
    assert generator.source_group({"hash": "chunk1"}) == generator.source_group({"hash": "chunk2"})


def test_connected_duplicate_groups_and_calibration():
    rows = [sample(f"doc{i}", f"Answer {i}") for i in range(200)]
    rows += [
        sample("doc0", "Micro QA derivative", "version"),
        sample("alias", " answer 0 "),
        sample("alias", "More content"),
        sample("calibration", "Cutoff", "cybersec", _train_only="true"),
    ]
    generator.RNG.seed(3407)
    train, val = generator.split_groups(copy.deepcopy(rows))
    assert not any(generator.audit_splits(train, val).values())
    assert not any(s.get("_train_only") for s in val)
    lookup = {s["output"]: side for side, split in (("train", train), ("val", val)) for s in split}
    assert (
        lookup["Answer 0"]
        == lookup["Micro QA derivative"]
        == lookup[" answer 0 "]
        == lookup["More content"]
    )
    generator.RNG.seed(3407)
    assert (train, val) == generator.split_groups(copy.deepcopy(rows))


def test_unsafe_and_unsplittable_data_fails():
    row = sample("a", "answer")
    with pytest.raises(ValueError, match="Unsafe"):
        generator.audit_splits([row], [row])
    with pytest.raises(ValueError, match="nonempty"):
        generator.split_groups([row])


def test_training_boost_preserves_unique_examples():
    generator.RNG.seed(3407)
    general = [sample(str(i), str(i)) for i in range(100)]
    versions = [sample("v", "Version", "version")]
    cyber = [sample("c", "Cyber", "cybersec")]
    boosted_v, boosted_c = generator._stratify_and_mix(general, versions, cyber)
    assert versions[0] in boosted_v and len(boosted_v) > len(versions)
    assert cyber[0] in boosted_c and len(boosted_c) > len(cyber)


def test_regeneration_determinism_schema_and_snapshots(tmp_path):
    source = tmp_path / "chunks.jsonl"
    chunks = [
        {
            "content": f"Unique documentation {i}: " + "some relevant Rust details " * 5,
            "source_type": "official_docs",
            "library": "book",
            "version": "2021",
            "url": f"https://example.org/{i}",
            "title": f"Topic {i}",
        }
        for i in range(200)
    ]
    source.write_text("\n".join(json.dumps(c) for c in chunks))
    output = tmp_path / "datasets"
    first = generator.generate_dataset(source, output, "2026-10-13")
    original = (output / "train.jsonl").read_bytes()
    second = generator.generate_dataset(source, output, "2026-10-13")
    assert first == second
    assert (output / "train.jsonl").read_bytes() == original
    assert list((output / "backups").glob("*/train.jsonl"))
    assert not any(first["checks"].values())
    assert not any(generator.verify_dataset(output).values())
    for name in ("train", "val"):
        rows = [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]
        assert all(set(row) == {"instruction", "input", "output"} for row in rows)
        provenance = (output / f"{name}.provenance.jsonl").read_text().splitlines()
        assert len(rows) == len(provenance)
    assert not list(tmp_path.glob("rust-lora-datasets-*"))
    with (output / "val.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        generator.verify_dataset(output)
