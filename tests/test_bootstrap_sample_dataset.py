from pathlib import Path

from scripts.ingest.bootstrap_sample_dataset import bootstrap_sample_dataset


def test_bootstrap_sample_dataset_copies_curated_samples(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample_data"
    (sample_root / "거주자증명서").mkdir(parents=True)
    (sample_root / "국내원천소득 제한세율").mkdir(parents=True)
    (sample_root / "아포스티유 샘플").mkdir(parents=True)

    for relative_path in [
        "거주자증명서/4.jpg",
        "거주자증명서/5.jpg",
        "거주자증명서/6.jpg",
        "거주자증명서/미국 TREASURY주.png",
        "거주자증명서/2.pdf",
        "국내원천소득 제한세율/국내원천소득 제한세율 적용신청서-1.png",
        "국내원천소득 제한세율/국내원천소득 제한세율 적용신청서-2.png",
        "국내원천소득 제한세율/원본 샘플.pdf",
        "아포스티유 샘플/미국 california 주.png",
        "아포스티유 샘플/미국 michigan 주.jpg",
        "아포스티유 샘플/미국 california 주2.jpg",
    ]:
        path = sample_root / relative_path
        path.write_bytes(b"sample")

    copied = bootstrap_sample_dataset(sample_root, tmp_path / "raw")

    assert len(copied) == 11
    assert (
        tmp_path / "raw" / "test" / "residency_certificate" / "residency_maria_chen_001.png"
    ).exists()
    assert (tmp_path / "raw" / "test" / "withholding_tax_form" / "withholding_pdf_001.pdf").exists()
    assert (tmp_path / "raw" / "test" / "apostille" / "apostille_north_carolina_001.jpg").exists()
