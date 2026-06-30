from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trained PaddleOCR recognizers into inference directories."
    )
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=Path("data/training/recognizer"),
    )
    parser.add_argument(
        "--paddleocr-home",
        type=Path,
        default=Path("PaddleOCR"),
    )
    parser.add_argument(
        "--checkpoint-subdir",
        default="model_output/best_accuracy",
        help="Checkpoint directory under each field-group directory.",
    )
    parser.add_argument(
        "--output-subdir",
        default="inference",
        help="Export output directory under each field-group directory.",
    )
    parser.add_argument("--field-group", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def render_export_command(
    plan_path: Path,
    paddleocr_home: Path,
    *,
    checkpoint_subdir: str,
    output_subdir: str,
) -> str:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    settings = plan["settings"]
    group_root = plan_path.parent
    export_py = paddleocr_home / "tools" / "export_model.py"
    checkpoint_dir = (group_root / checkpoint_subdir).resolve()
    output_dir = (group_root / output_subdir).resolve()
    dict_path = Path(settings["dictionary_path"]).resolve()
    base_config = str((paddleocr_home / settings["base_config"]).resolve())
    return (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(export_py))} "
        f"-c {shlex.quote(str(base_config))} "
        f"-o Global.use_gpu=False "
        f"Global.pretrained_model={shlex.quote(str(checkpoint_dir))} "
        f"Global.save_inference_dir={shlex.quote(str(output_dir))} "
        f"Global.character_dict_path={shlex.quote(str(dict_path))}"
    )


def normalize_inference_dir(output_dir: Path) -> None:
    legacy_model = output_dir / "inference.pdmodel"
    pir_json = output_dir / "inference.json"
    if legacy_model.exists() and pir_json.exists():
        pir_json.unlink()


def export_recognizer_inference(
    plan_root: Path,
    paddleocr_home: Path,
    *,
    checkpoint_subdir: str = "model_output/best_accuracy",
    output_subdir: str = "inference",
    field_groups: list[str] | None = None,
    execute: bool = False,
) -> dict[str, str]:
    commands: dict[str, str] = {}
    groups = field_groups or [
        path.name
        for path in sorted(plan_root.iterdir())
        if path.is_dir() and (path / "plan.json").exists()
    ]
    export_py = paddleocr_home / "tools" / "export_model.py"
    if execute and not export_py.exists():
        raise FileNotFoundError(f"Missing PaddleOCR export entrypoint: {export_py}")

    for field_group in groups:
        plan_path = plan_root / field_group / "plan.json"
        if not plan_path.exists():
            continue
        command = render_export_command(
            plan_path,
            paddleocr_home,
            checkpoint_subdir=checkpoint_subdir,
            output_subdir=output_subdir,
        )
        commands[field_group] = command
        if execute:
            subprocess.run(command, shell=True, check=True)
            normalize_inference_dir(plan_root / field_group / output_subdir)
    return commands


def main() -> None:
    args = parse_args()
    commands = export_recognizer_inference(
        args.plan_root,
        args.paddleocr_home,
        checkpoint_subdir=args.checkpoint_subdir,
        output_subdir=args.output_subdir,
        field_groups=args.field_group or None,
        execute=args.execute,
    )
    print(json.dumps(commands, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
