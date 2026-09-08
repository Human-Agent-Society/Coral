#!/opt/venvs/train/bin/python
from __future__ import annotations

import json
from pathlib import Path

from common import PROMPT_PREFIX

SOURCE = Path("/app/data/train.jsonl")
OUTPUT = Path("/app/results/answer-sft")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "answer_sft.jsonl"
    count = 0
    with SOURCE.open() as source, target.open("w") as destination:
        for line in source:
            row = json.loads(line)
            example = {
                "messages": [
                    {"role": "user", "content": PROMPT_PREFIX + row["problem"]},
                    {"role": "assistant", "content": f"Final answer: \\boxed{{{row['answer']}}}"},
                ]
            }
            destination.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    info = {
        "answer_sft": {
            "file_name": target.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role", "content_tag": "content",
                "user_tag": "user", "assistant_tag": "assistant",
            },
        }
    }
    (OUTPUT / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"examples": count, "path": str(target)}))


if __name__ == "__main__":
    main()
