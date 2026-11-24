from openpi.training import config
from dataclasses import replace

def build_train_config():
    base = config.get_config("pi05_libero")
    model =  build_lora_model_config(base.model)
    return replace(
        base,
        model=model,
        freeze_filter = model.get_freeze_filter(),
        ema_decay = None,
        batch_size = 1, #模型每个batch跑的样本数，后面正式运行要改
        num_train_steps = 2, #训练步数，后面正式运行要改
        exp_name = "lora_smoke_01",
        wandb_enabled = False,
        log_interval = 1, #多少步打印一次loss
        )

def build_lora_model_config(base_model):
    return replace(
        base_model,
        paligemma_variant = "gemma_2b_lora",
        action_expert_variant = "gemma_300m_lora",
    )

