#!/usr/bin/env python3
"""Copy SPIB config inputs to node-local storage and rewrite their paths."""

from __future__ import annotations

import argparse
import configparser
import shutil
from pathlib import Path


LIST_KEYS = ("traj_data", "initial_labels", "traj_weights")
SINGLE_KEYS = ("data_mean", "data_std")


def split_paths(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_input(raw_path: str, code_root: Path, data_root: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        choices = (candidate,)
    else:
        choices = (code_root / candidate, data_root / candidate)
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    searched = ", ".join(str(path) for path in choices)
    raise FileNotFoundError(f"Cannot find SPIB input {raw_path!r}; searched: {searched}")


def copy_input(source: Path, destination_root: Path, index: int) -> Path:
    destination = destination_root / f"input-{index:03d}-{source.name}"
    shutil.copy2(source, destination)
    print(f"staged_input[{index}]={source} -> {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--scratch-data", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    args = parser.parse_args()

    config = configparser.ConfigParser(allow_no_value=True)
    if not config.read(args.config):
        raise FileNotFoundError(f"Cannot read config: {args.config}")
    if not config.has_section("Data"):
        raise ValueError("SPIB config has no [Data] section")

    args.scratch_data.mkdir(parents=True, exist_ok=True)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    input_index = 0

    for key in LIST_KEYS:
        value = config.get("Data", key, fallback=None)
        if value is None:
            continue
        rewritten = []
        for raw_path in split_paths(value):
            source = resolve_input(raw_path, args.code_root, args.data_root)
            rewritten.append(str(copy_input(source, args.scratch_data, input_index)))
            input_index += 1
        if rewritten:
            config.set("Data", key, "[" + ", ".join(rewritten) + "]")

    for key in SINGLE_KEYS:
        value = config.get("Data", key, fallback=None)
        if value is None or not value.strip():
            continue
        source = resolve_input(value.strip(), args.code_root, args.data_root)
        config.set("Data", key, str(copy_input(source, args.scratch_data, input_index)))
        input_index += 1

    if input_index == 0:
        raise ValueError("No input files were declared in the SPIB [Data] section")
    with args.output_config.open("w", encoding="utf-8") as handle:
        config.write(handle)
    print(f"staged_config={args.output_config}", flush=True)


if __name__ == "__main__":
    main()
