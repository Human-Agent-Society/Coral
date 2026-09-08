#!/opt/venvs/train/bin/python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="/app/submission")
    args = parser.parse_args()
    output = Path(args.output)
    if output.resolve() in (Path("/app/models/student").resolve(), Path("/app/models/teacher").resolve()):
        raise ValueError("refusing to overwrite a fixed model")
    output.mkdir(parents=True, exist_ok=True)
    for child in output.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    model = AutoModelForCausalLM.from_pretrained(
        "/app/models/student", dtype="auto", device_map={"": 0}, local_files_only=True
    )
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained("/app/models/student", local_files_only=True).save_pretrained(output)
    if (output / "adapter_config.json").exists():
        raise RuntimeError("merge unexpectedly produced an adapter-only checkpoint")
    print(f"merged standalone model: {output}")


if __name__ == "__main__":
    main()
