import os
import json
import argparse
from eval_utils import evaluate
import tqdm


def main(args):
    predictions = []
    for file in tqdm.tqdm(os.listdir(args.log_dir)):
        try:
            pred = json.load(open(os.path.join(args.log_dir, file), "r"))[-1]
            if "answer" in pred:
                predictions.append(pred)
        except Exception as e:
            os.remove(os.path.join(args.log_dir, file))
            print(f"Error loading {file}: {e}")

    metric = "soft match"
    # metric = "strict match"
    acc, failure_question_ids = evaluate(predictions, metric)
    print(f"{metric} accuracy of {len(predictions)} samples:", "{:.2%}".format(acc))

    if args.log_failure:
        with open(args.log_failure, "w+") as f:
            json.dump(failure_question_ids, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluation of LLM-TPC")
    parser.add_argument("--log_dir", type=str, help="path to log files")
    parser.add_argument(
        "--log_failure", "-f", type=str, help="path to failure question ids"
    )
    args = parser.parse_args()
    main(args)
