import collections
import json
from pprint import pprint
from typing import List, Optional

import numpy as np
from scipy.stats import hmean

from util.globals import *


def main(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    for run_dir in (RESULTS_DIR / dir_name if not abs_path else dir_name).iterdir():
        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("*case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")

            case_id = data["case_id"]
            if first_n_cases is not None and case_id >= first_n_cases:
                break

            if "time" in data:
                cur_sum["time"].append(data["time"])

            for prefix in ["pre", "post"]:
                # Probability metrics for which new should be lower (better) than true  #新改的概率大于旧改的概率，就实现了success，但这里是反过来的？？不知道为什么，可能由于是负值？
                for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue

                    sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"

                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] > x["target_new"]   #新改的概率大于旧改的概率，就实现了success，但这里是反过来的？？不知道为什么，可能由于是负值？
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # Probability metrics for which true should be lower (better) than new
                sum_key_discrete = f"{prefix}_neighborhood_success"
                sum_key_cont = f"{prefix}_neighborhood_diff"
                key = "neighborhood_prompts_probs"
                if prefix in data and key in data[prefix]:
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] < x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # Accuracy-based evaluation metrics
                for key in ["rewrite", "paraphrase", "neighborhood"]:
                    sum_key = f"{prefix}_{key}_acc"
                    key = f"{key}_prompts_correct"

                    if prefix not in data or key not in data[prefix]:
                        continue

                    cur_sum[sum_key].append(np.mean(data[prefix][key]))

                # Generation metrics that can be directly averaged
                for key in ["ngram_entropy", "reference_score", "essence_score"]:
                    if prefix in data and key in data[prefix]:
                        cur_sum[f"{prefix}_{key}"].append(data[prefix][key])

        if len(cur_sum) == 0:
            continue

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }

        uncompressed.append(dict(cur_sum, **metadata))

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                # Constant multiplication scales linearly with mean and stddev
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

        for prefix in ["pre", "post"]:
            for k_efficacy, k_generalization, k_specificity in [
                (
                    f"{prefix}_rewrite_success",
                    f"{prefix}_paraphrase_success",
                    f"{prefix}_neighborhood_success",
                ),
                # (
                #     f"{prefix}_rewrite_acc",
                #     f"{prefix}_paraphrase_acc",
                #     f"{prefix}_neighborhood_acc",
                # ),
            ]:
                if all(k in cur_sum for k in [k_efficacy, k_generalization, k_specificity]):
                    hmean_list = [
                        cur_sum[k_efficacy][0],   #所以后面那个指标[1]应该就是方差
                        cur_sum[k_generalization][0],
                        cur_sum[k_specificity][0],
                    ]

                    # if f"{prefix}_ngram_entropy" in cur_sum:
                    #     hmean_list.append(2 ** (cur_sum[f"{prefix}_ngram_entropy"][0] / 100))
                    # if f"{prefix}_reference_score" in cur_sum:
                    #     hmean_list.append(cur_sum[f"{prefix}_reference_score"][0])

                    cur_sum[f"{prefix}_score"] = (hmean(hmean_list), np.nan)   #所以后面那个指标应该就是方差,score就是三个指标的和
                    break

        cur_sum.update(metadata)
        pprint(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    args = parser.parse_args()

    main(
        args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
    )



# import json
# json_file = "E:/NLP/LLM/Unlearning/memit-main/data/zsre_mend_eval.json"
#
# # 打开 JSON 文件并解析
# with open(json_file, "r") as file:
#     data = json.load(file)
#
# # 现在 data 变量包含了 JSON 数据的 Python 字典表示
# # 你可以访问和操作 data 中的数据
# print(len(data))

# import torch
# remote_url = "https://memit.baulab.info/data/dsets/zsre_mend_train.json"
# cf_loc = "E:/NLP/LLM/Unlearning/memit-main/data/zsre_mend_train.json"
# torch.hub.download_url_to_file(remote_url, cf_loc)



# import json
#
# # 指定输入 JSON 文件名和输出 JSON 文件名
# input_file_name = "E:/NLP/LLM/Unlearning/memit-main/data/multi_counterfact_1000_rephrase__.json"
# output_file_name = "E:/NLP/LLM/Unlearning/memit-main/data/multi_counterfact_1000_rephrase___.json"
#
# # 打开输入文件并读取数据
# with open(input_file_name, "r") as input_file:
#     data = json.load(input_file)
#
# print(len(data))
#
# # 提取前 1000 条数据
# first_1000_records = data[413:]
#
# # 将前 1000 条数据保存到输出文件
# with open(output_file_name, "w") as output_file:
#     json.dump(first_1000_records, output_file)

