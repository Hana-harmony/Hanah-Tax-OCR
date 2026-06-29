from pathlib import Path

from scripts.ingest.build_staging_manifest import build_manifest, write_jsonl


def test_build_manifest_indexes_supported_files(tmp_path: Path) -> None:
    source = tmp_path / "staging" / "residency_certificate"
    source.mkdir(parents=True)
    (source / "sample.png").write_bytes(b"fake")
    (source / "ignore.txt").write_text("x", encoding="utf-8")

    entries = build_manifest(tmp_path / "staging")

    assert len(entries) == 1
    assert entries[0]["filename"] == "sample.png"

    output = tmp_path / "manifest.jsonl"
    write_jsonl(output, entries)
    assert output.exists()
