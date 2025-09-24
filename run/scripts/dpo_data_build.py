import sys
from tqdm import tqdm
import json
import os
import argparse
import sys
from eval_utils import soft_match

original_dir = ""
output_dir = ""
original_llm_tpc_dir = ""
original_llm_tpc_files = []


def parse_filename(filename):
    # scene0704_00_220602033363-2024-10-17-22:13:03.json
    return filename[:12], filename[13:25]


def get_question_id(filename):
    return filename[13:25]


GET_INSTRUCTION_FROM_FILE = True
SYSTEM_INSTRUCT = ""


def get_system_instruction(question_file):
    global GET_INSTRUCTION_FROM_FILE
    global SYSTEM_INSTRUCT
    if GET_INSTRUCTION_FROM_FILE:
        question = json.load(
            open(os.path.join(original_llm_tpc_dir, question_file), "r")
        )
        SYSTEM_INSTRUCT = "\n".join([i["content"] for i in question[:14]])
        GET_INSTRUCTION_FROM_FILE = False


def get_instruction(question_data):
    human_msg = question_data[14]["content"]

    return [
        {"from": "system", "value": SYSTEM_INSTRUCT},
        {"from": "human", "value": human_msg},
    ]


def handle_question(question_file):
    try:
        data = json.load(open(os.path.join(original_llm_tpc_dir, question_file), "r"))
    except:
        tqdm.write(f"error in loading {question_file}")
        return []

    pred = data[-1]
    acc = False
    if "answer" in pred:
        pred_answer = pred["answer"]
        gt_ansmer = pred["gt_answer"]
        acc = soft_match(pred_answer, gt_ansmer)
    else:
        tqdm.write(f"no answer in {question_file}")
        return []

    if not acc:
        return []

    # reach max try
    if len(data) >= 29:
        tqdm.write(f"reach max try: {question_file}")
        return []

    # success in first try
    if len(data) <= 19:
        # tqdm.write(f"success in first try: {question_file}")
        return []

    success = len(data) - 4

    dpo_data = []
    instruction = get_instruction(data)
    i = 15
    while i < success:
        dpo_data.append(
            {
                "conversations": instruction,
                "chosen": {"from": "gpt", "value": data[success]["content"]},
                "rejected": {"from": "gpt", "value": data[i]["content"]},
            }
        )
        i += 2

    return dpo_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dpo data build of LLM-TPC")
    parser.add_argument("--original_dir", type=str)
    args = parser.parse_args()

    original_dir = args.original_dir

    original_llm_tpc_dir = os.path.join(original_dir, "llm-tpc")
    original_llm_tpc_files = sorted(os.listdir(original_llm_tpc_dir))

    output_file = os.path.join(original_dir, "dpo_data_train_all_2.json")

    get_system_instruction(original_llm_tpc_files[0])

    dpo_datas = []
    for question_file in tqdm(original_llm_tpc_files):
        question_to_dpo_data = handle_question(question_file)
        dpo_datas.extend(question_to_dpo_data)

    with open(output_file, "w+", encoding="utf-8") as f:
        json.dump(dpo_datas, f, ensure_ascii=False, indent=2)

    print(f"{len(dpo_datas)} dpo data has been built")
