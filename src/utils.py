import pandas as pd
import json


def load_tsv(file_path):
    return pd.read_csv(file_path, sep="\t")


# MARCのlabel処理 [ 1 → negative, 2 → neutral, 3以上 → positive ]
def make_amazon_reviews_3labels(data_path):
    names = ["/train.jsonl", "/test.jsonl", "/valid.jsonl"]

    for n in names:
        data = []
        with open(data_path + n, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Error decoding JSON for line: {line}")
                    continue

            for item in data:
                if item["label"] <= 1:
                    item["label_text"] = "negative"
                elif item["label"] >= 3:
                    item["label_text"] = "positive"
                elif item["label"] == 2:
                    item["label_text"] = "neutral"
                del item["id"]
                del item["label"]
                item["label"] = item["label_text"]
                del item["label_text"]
            filtered_data = [item for item in data]
            # if item["label"] != "neutral" 追加でポジネガになる

        with open(data_path + n, "w", encoding="utf-8") as f:
            for entry in filtered_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")


# jsonlファイルをtsvファイルに変換
def jsonl_to_tsv(jsonl_file_path, tsv_file_path):
    with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:
        # 最初の行からキーを取得
        first_line = jsonl_file.readline()
        keys = list(json.loads(first_line).keys())

        # ファイルポインタを先頭に戻す
        jsonl_file.seek(0)

        with open(tsv_file_path, "w", encoding="utf-8", newline="") as tsv_file:
            # ヘッダー行を書き込む
            tsv_file.write("\t".join(keys) + "\n")

            # JSONLファイルの各行を処理
            for line in jsonl_file:
                json_obj = json.loads(line)
                values = [str(json_obj[key]) for key in keys]
                tsv_file.write("\t".join(values) + "\n")
