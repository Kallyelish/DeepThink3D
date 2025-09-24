import sys
from tqdm import tqdm
import json
import os
import argparse
import sys
from eval_utils import soft_match
import copy

original_dir = ""
output_dir = ""
original_llm_tpc_dir = ""
original_llm_tpc_files = []


def parse_filename(filename):
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


def handle_question(question_file):
    global SYSTEM_INSTRUCT
    data = json.load(open(os.path.join(original_llm_tpc_dir, question_file), "r"))
    pred = data[-1]
    acc = False
    if "answer" in pred:
        pred_answer = pred["answer"]
        gt_ansmer = pred["gt_answer"]
        acc = soft_match(pred_answer, gt_ansmer)
    else:
        print(f"no answer in {question_file}")
        return None, []

    if not acc:
        return None, []

    # reach max try
    if len(data) >= 29:
        print(f"reach max try: {question_file}")
        return None, []

    # # auccess in first try
    # if question_begin - question_end == 1:
    #     print(f"success in first try: {question_file}")
    #     return []

    sft_data_with_history = []
    human_instruct = data[14]["content"]
    i = 15
    success = len(data) - 4
    sft_data = {
        "instruction": human_instruct,
        "output": data[success]["content"],
        "system": SYSTEM_INSTRUCT,
    }
    history_data = []
    while i <= success:
        input = data[i - 1]["content"]
        response = data[i]["content"]
        sft_data_with_history.append(
            {
                "instruction": input,
                "output": response,
                "system": SYSTEM_INSTRUCT,
                "history": copy.deepcopy(history_data),
            }
        )
        i += 2
        history_data.append([input, response])

    return sft_data, sft_data_with_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dpo data build of LLM-TPC")
    parser.add_argument("--original_dir", type=str)
    args = parser.parse_args()

    original_dir = args.original_dir

    original_llm_tpc_dir = os.path.join(original_dir, "llm-tpc")
    original_llm_tpc_files = sorted(os.listdir(original_llm_tpc_dir))

    output_file = os.path.join(original_dir, "sft_data_test_all.json")
    output_file_with_history = os.path.join(
        original_dir, "sft_data_with_history_test_all.json"
    )

    get_system_instruction(original_llm_tpc_files[0])

    sft_datas = []
    sft_datas_with_history = []
    for question_file in tqdm(original_llm_tpc_files):
        question_to_sft_data, question_to_sft_data_with_history = handle_question(
            question_file
        )
        if question_to_sft_data is not None:
            sft_datas.append(question_to_sft_data)
            sft_datas_with_history.extend(question_to_sft_data_with_history)

    with open(output_file, "w+", encoding="utf-8") as f:
        json.dump(sft_datas, f, ensure_ascii=False, indent=2)

    print(f"{len(sft_datas)} sft data has been built")

    with open(output_file_with_history, "w+", encoding="utf-8") as f:
        json.dump(sft_datas_with_history, f, ensure_ascii=False, indent=2)

    print(f"{len(sft_datas_with_history)} sft data with history has been built")
