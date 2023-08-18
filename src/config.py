import torch
import warnings

warnings.filterwarnings("ignore")

# path
DIR = "multi_task_learning_with_t5"
MARC_DIR = DIR + "/data/MARC"
AMCD_DIR = DIR + "/data/AMCD"
MODEL_DIR = DIR + "/models"
CHECKPOINT = DIR + "/checkpoint"

# model params　train_batch_size,eval_batch_size 8
USE_GPU = torch.cuda.is_available()
PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"

# 変更：
args_dict = dict(
    data_dir=DIR + "/data",  # データセットのディレクトリ
    model_name_or_path=PRETRAINED_MODEL_NAME,
    tokenizer_name_or_path=PRETRAINED_MODEL_NAME,
    learning_rate=3e-4,
    betas=(0.9, 0.999),
    adam_epsilon=1e-8,
    weight_decay=0.0,
    warmup_steps=0,
    gradient_accumulation_steps=1,
    max_input_length=512,
    max_target_length=8,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=4,
    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    fp_16=False,
    max_grad_norm=1.0,
    seed=12,
)
