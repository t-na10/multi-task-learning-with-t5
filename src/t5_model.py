from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import sys
import warnings

sys.path.append("multi_task_learning_with_t5")
warnings.filterwarnings("ignore")
import src.config as config


class T5Model:
    def __init__(self, multi_task=False, task_weights=None, experiment_name=None):
        self.tokenizer = T5Tokenizer.from_pretrained(
            config.PRETRAINED_MODEL_NAME, use_legacy_mode=False
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.PRETRAINED_MODEL_NAME
        )
        if experiment_name:
            self.writer = SummaryWriter("runs/" + experiment_name)
        else:
            self.writer = SummaryWriter()
        self.multi_task = multi_task

        if multi_task:
            self.task_weights = (
                task_weights if task_weights else {"MARC": 0.7, "AMCD": 0.3}
            )

        # GPUの使用
        # if config.USE_GPU:
        #     if torch.cuda.device_count() > 1:
        #         print(f"Using {torch.cuda.device_count()} GPUs")
        #         self.model = torch.nn.DataParallel(self.model)
        #     self.model.cuda()

        if config.USE_GPU:
            self.model.cuda()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.args_dict["learning_rate"],
            betas=config.args_dict["betas"],
            eps=config.args_dict["adam_epsilon"],
            weight_decay=config.args_dict["weight_decay"],
        )

    # multi-task時のみ使用
    # タスクに応じた重みを返す
    def get_task_weights(self, task_type):
        return self.task_weights.get(task_type, 1.0)

    def train(self, train_dataset, valid_dataset=None):
        train_loader = DataLoader(
            train_dataset, batch_size=config.args_dict["train_batch_size"], shuffle=True
        )

        for epoch in range(config.args_dict["num_train_epochs"]):
            self.model.train()
            total_loss = 0

            # tqdmを使用してプログレスバーを表示
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            for batch_idx, batch in enumerate(progress_bar):
                inputs = self.tokenizer(
                    batch["input_text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.args_dict["max_input_length"],
                )
                targets = self.tokenizer(
                    batch["target_text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.args_dict["max_target_length"],
                )
                # GPUの使用
                if config.USE_GPU:
                    inputs = {key: tensor.cuda() for key, tensor in inputs.items()}
                    targets = {key: tensor.cuda() for key, tensor in targets.items()}
                # Forward
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=targets["input_ids"],
                )
                # 損失
                loss = outputs.loss
                # バッチ内の損失を平均化（DataParallelを使用している場合）
                # if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                #     loss = loss.mean()

                # プログレスバーに損失情報を追加
                progress_bar.set_postfix({"loss": loss.item()})
                # Multi-task学習時の損失計算
                if self.multi_task:
                    task_weights = self.get_task_weights(batch["task_type"][0])
                    loss = loss * task_weights
                # Backward
                loss.backward()
                # ログに損失を追加
                total_loss += loss.item()
                # 勾配の累積
                if (batch_idx + 1) % config.args_dict[
                    "gradient_accumulation_steps"
                ] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # エポックごとの平均損失を表示
            avg_train_loss = total_loss / len(train_loader)

            if valid_dataset is not None:
                eval_metrics = self.validation(valid_dataset)

                print(
                    f'Epoch [{epoch+1}/{config.args_dict["num_train_epochs"]}], Loss: {avg_train_loss:.4f}, val_loss: {eval_metrics["loss"]:.4f}, val_acc: {eval_metrics["accuracy"]:.4f}'
                )
                # TensorBoardに訓練損失を記録
                self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
                # TensorBoardに検証損失を記録
                self.writer.add_scalar("Loss/val", eval_metrics["loss"], epoch)
                # TensorBoardに検証精度を記録
                self.writer.add_scalar("Accuracy/val", eval_metrics["accuracy"], epoch)
            else:
                print(
                    f'Epoch [{epoch+1}/{config.args_dict["num_train_epochs"]}], Loss: {avg_train_loss:.4f}'
                )
                # TensorBoardに訓練損失を記録
                self.writer.add_scalar("Loss/train", avg_train_loss, epoch)

    def validation(self, valid_dataset):
        valid_datasets = DataLoader(
            valid_dataset, batch_size=config.args_dict["eval_batch_size"], shuffle=False
        )
        self.model.eval()

        predictions = []
        true_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in valid_datasets:
                inputs = self.tokenizer(
                    batch["input_text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.args_dict["max_input_length"],
                )
                targets = self.tokenizer(
                    batch["target_text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.args_dict["max_target_length"],
                )

                if config.USE_GPU:
                    inputs = {key: tensor.cuda() for key, tensor in inputs.items()}
                    targets = {key: tensor.cuda() for key, tensor in targets.items()}

                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=targets["input_ids"],
                )

                loss = outputs.loss

                # バッチ内の損失を平均化（DataParallelを使用している場合）
                # if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                #     loss = loss.mean()

                total_loss += loss.item()

                dec = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in outputs.logits.argmax(dim=-1)
                ]
                target = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in targets["input_ids"]
                ]

                # 動作確認
                # print(f"target:{target}, prediction:{dec}")

                predictions.extend(dec)
                true_labels.extend(target)

        # Compute the evaluation metrics directly on the decoded texts
        avg_loss = total_loss / len(valid_datasets)
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average="macro")
        f1_micro = f1_score(true_labels, predictions, average="micro")
        precision_macro = precision_score(true_labels, predictions, average="macro")
        precision_micro = precision_score(true_labels, predictions, average="micro")
        recall_macro = recall_score(true_labels, predictions, average="macro")
        recall_micro = recall_score(true_labels, predictions, average="micro")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_score_macro": f1_macro,
            "f1_score_micro": f1_micro,
            "precision_macro": precision_macro,
            "precision_micro": precision_micro,
            "recall_macro": recall_macro,
            "recall_micro": recall_micro,
        }

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.model = T5ForConditionalGeneration.from_pretrained(path)

    def evaluate(self, test_dataset):
        test_loader = DataLoader(
            test_dataset, batch_size=config.args_dict["eval_batch_size"], shuffle=False
        )
        self.model.eval()

        predictions = []
        true_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs = self.tokenizer(
                    batch["input_text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.args_dict["max_input_length"],
                )
                targets = self.tokenizer(
                    batch["target_text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.args_dict["max_target_length"],
                )

                if config.USE_GPU:
                    inputs = {key: tensor.cuda() for key, tensor in inputs.items()}
                    targets = {key: tensor.cuda() for key, tensor in targets.items()}

                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=targets["input_ids"],
                )

                loss = outputs.loss

                # バッチ内の損失を平均化（DataParallelを使用している場合）
                # if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                #     loss = loss.mean()

                total_loss += loss.item()

                dec = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in outputs.logits.argmax(dim=-1)
                ]
                target = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in targets["input_ids"]
                ]

                # 動作確認
                # print(f"target:{target}, prediction:{dec}")

                predictions.extend(dec)
                true_labels.extend(target)

        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average="macro")
        f1_micro = f1_score(true_labels, predictions, average="micro")
        precision_macro = precision_score(true_labels, predictions, average="macro")
        precision_micro = precision_score(true_labels, predictions, average="micro")
        recall_macro = recall_score(true_labels, predictions, average="macro")
        recall_micro = recall_score(true_labels, predictions, average="micro")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_score_macro": f1_macro,
            "f1_score_micro": f1_micro,
            "precision_macro": precision_macro,
            "precision_micro": precision_micro,
            "recall_macro": recall_macro,
            "recall_micro": recall_micro,
        }
