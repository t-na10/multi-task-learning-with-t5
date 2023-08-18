import os
import sys
import shutil
import subprocess
from utils import jsonl_to_tsv, make_amazon_reviews_3labels

sys.path.append("multi_task_learning_with_t5")

from src.config import MARC_DIR, AMCD_DIR

# 以下のリンクをクローンする
# https://github.com/amazon-science/amazon-multilingual-counterfactual-dataset/tree/main/data
# クローンからコピー
if os.path.exists(AMCD_DIR) is False:
    new_path = AMCD_DIR
    original_path = "your cloned path"
    names = ["JP_train.tsv", "JP_valid.tsv", "JP_test.tsv"]
    new_names = ["train.tsv", "valid.tsv", "test.tsv"]
    os.mkdir(new_path)
    for n, nn in zip(names, new_names):
        shutil.copy(original_path + n, new_path + "/" + nn)

else:
    print("AMCD dataset is already exist!")

# MARCをダウンロード
if os.path.exists(MARC_DIR) is False:
    new_path = MARC_DIR
    url = "https://huggingface.co/datasets/SetFit/amazon_reviews_multi_ja/resolve/main"
    names = ["train.jsonl", "test.jsonl", "validation.jsonl"]
    new_names = ["train.jsonl", "test.jsonl", "valid.jsonl"]
    os.mkdir(new_path)
    for n, nn in zip(names, new_names):
        try:
            cmd = ["wget", "-q", "-O", new_path + "/" + nn, url + "/" + n]
            result = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(e)

    make_amazon_reviews_3labels(new_path)

else:
    print("amazon_reviews_ja is already exist!")

if os.path.exists(MARC_DIR + "/train.tsv") is False:
    jsonl_file_path = ["/train.jsonl", "/valid.jsonl", "/test.jsonl"]
    tsv_file_path = ["/train.tsv", "/valid.tsv", "/test.tsv"]
    for j, t in zip(jsonl_file_path, tsv_file_path):
        jsonl_to_tsv(MARC_DIR + j, MARC_DIR + t)
