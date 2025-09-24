import sys
from tqdm import tqdm
import json
import os
import argparse
import sys
from eval_utils import soft_match

base_dir = ""
stage1_dir = "train_all_8b_sft_step300"
stage2_dir = "train_all_8b_sft_step300_stage2"


def get_question_id(filename):
    return filename[13:25]


GET_INSTRUCTION_FROM_FILE = True
SYSTEM_INSTRUCT = ""


def get_system_instruction(question_file):
    global GET_INSTRUCTION_FROM_FILE
    global SYSTEM_INSTRUCT
    if GET_INSTRUCTION_FROM_FILE:
        question = json.load(
            open(os.path.join(base_dir, stage2_dir, "llm-tpc", question_file), "r")
        )
        SYSTEM_INSTRUCT = "\n".join([i["content"] for i in question[:14]])
        GET_INSTRUCTION_FROM_FILE = False


def get_instruction(question_data):
    human_msg = question_data[14]["content"]

    return [
        {"from": "system", "value": SYSTEM_INSTRUCT},
        {"from": "human", "value": human_msg},
    ]


if __name__ == "__main__":
    stage1_files = os.listdir(os.path.join(base_dir, stage1_dir, "llm-tpc"))
    stage2_files = os.listdir(os.path.join(base_dir, stage2_dir, "llm-tpc"))

    stage1_ids = [get_question_id(question_file) for question_file in stage1_files]

    dpo_data = []
    error_files = []

    get_system_instruction(stage2_files[0])

    for stage2_file in tqdm(stage2_files):
        question_id = get_question_id(stage2_file)

        try:
            stage2_data = json.load(
                open(os.path.join(base_dir, stage2_dir, "llm-tpc", stage2_file), "r")
            )
        except:
            tqdm.write(f"error in loading {stage2_file}")
            error_files.append(stage2_file)
            continue

        pred = stage2_data[-1]
        acc = False
        if "answer" in pred:
            pred_answer = pred["answer"]
            gt_ansmer = pred["gt_answer"]
            acc = soft_match(pred_answer, gt_ansmer)
        else:
            tqdm.write(f"no answer in {stage2_file}")
            continue

        if not acc:
            continue

        idx = stage1_ids.index(question_id)
        stage1_file = stage1_files[idx]
        stage1_data = json.load(
            open(os.path.join(base_dir, stage1_dir, "llm-tpc", stage1_file), "r")
        )

        instruction = get_instruction(stage1_data)

        dpo_data.append(
            {
                "conversations": instruction,
                "chosen": {"from": "gpt", "value": stage2_data[-4]["content"]},
                "rejected": {
                    "from": "gpt",
                    "value": stage1_data[-4]["content"]
                    if len(stage1_data) < 30
                    else stage1_data[-5]["content"],
                },
            }
        )

    with open(
        os.path.join(base_dir, stage2_dir, "dpo_data_stage1_vs_stage2.json"),
        "w+",
        encoding="utf-8",
    ) as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)

    print(
        f"dpo_data_stage1_vs_stage2.json saved to {os.path.join(base_dir, stage2_dir, 'dpo_data_stage1_vs_stage2.json')}"
    )
    print(f"total data: {len(dpo_data)}")
    print(f"error files: {len(error_files)}")
