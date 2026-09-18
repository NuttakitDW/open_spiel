#!/usr/bin/env python3
"""Launch one bounded, durable local Hold'em training job on macOS."""

import argparse
import hashlib
import json
import os
import plistlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=int, required=True)
    parser.add_argument("--train-seconds", type=int, default=43200)
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--label", default="local.openspiel.hulhe.20260917")
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    if args.deadline_epoch <= 0 or args.train_seconds < 0 or args.hands < 1:
        parser.error("invalid deadline, training seconds, or hand count")

    repository = Path(__file__).resolve().parents[2]
    binary = (args.binary or repository / "build-hu-make/examples/hu_limit_bot").resolve()
    runner = repository / "open_spiel/examples/run_hu_limit_bot.sh"
    run_dir = args.run_dir.resolve()
    if not binary.is_file() or not runner.is_file():
        parser.error("bot binary or runner script is missing")
    target = f"gui/{os.getuid()}/{args.label}"
    if subprocess.run(["launchctl", "print", target], capture_output=True).returncode == 0:
        parser.error(f"launchd job already loaded: {target}")

    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_binary = run_dir / "hu_limit_bot"
    frozen_runner = run_dir / "run_hu_limit_bot.sh"
    if (run_dir / "run.status").exists() or (run_dir / "current.chk").exists():
        parser.error("run directory already contains a job; choose a fresh directory")
    shutil.copy2(binary, frozen_binary)
    shutil.copy2(runner, frozen_runner)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "deadline_epoch": args.deadline_epoch,
        "train_seconds": args.train_seconds,
        "evaluation_hands_per_baseline": args.hands,
        "binary_sha256": sha256(frozen_binary),
        "runner_sha256": sha256(frozen_runner),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    plist_path = run_dir / "job.plist"
    job = {
        "Label": args.label,
        "ProgramArguments": [
            "/usr/bin/caffeinate", "-is", "/bin/bash", str(frozen_runner),
            str(frozen_binary), str(run_dir), str(args.deadline_epoch),
            str(args.train_seconds), str(args.hands),
        ],
        "RunAtLoad": True,
        "KeepAlive": False,
        "WorkingDirectory": str(repository),
        "StandardOutPath": str(run_dir / "stdout.log"),
        "StandardErrorPath": str(run_dir / "stderr.log"),
    }
    with plist_path.open("wb") as output:
        plistlib.dump(job, output)
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)], check=True)
    print(f"Launched {target}\nRun directory: {run_dir}\nBinary SHA256: {manifest['binary_sha256']}")


if __name__ == "__main__":
    main()
