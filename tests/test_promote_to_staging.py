from pathlib import Path

from scripts.ingest.promote_to_staging import promote_to_staging


def test_promote_to_staging_copies_supported_files(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    (source / "a.png").write_bytes(b"png")
    (source / "b.txt").write_text("ignore", encoding="utf-8")

    copied = promote_to_staging(source, "residency_certificate", tmp_path / "staging")

    assert len(copied) == 1
    assert copied[0].name == "a.png"
    assert copied[0].exists()
