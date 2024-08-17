import logging

import hydra
import numpy as np
import optuna
import torch
import torch.nn as nn
from hydra import compose, initialize
from optuna import Study
from optuna.trial import FrozenTrial

from config import Config
from dataloader import CustomDataLoader
from trainer import Trainer
from utils import init_logger


def init_seed(seed: int) -> None:
    torch.manual_seed(seed)  # PyTorchの乱数シードを固定
    torch.backends.cudnn.deterministic = True  # 計算の再現性を保証
    torch.backends.cudnn.benchmark = False  # 性能向上のための最適化を無効化
    np.random.seed(seed)  # NumPyのランダムシードを固定


def main(trial: FrozenTrial, cfg: Config) -> None:
    init_seed(cfg.seed)

    # 学習率と重み減衰の値を取得
    lr = round(trial.suggest_loguniform("lr", 1e-5, 1e-1), 6)
    betas = tuple(map(float, cfg.optimizer.betas))
    eps = round(trial.suggest_loguniform("eps", 1e-10, 1e-3), 6)
    weight_decay = round(trial.suggest_loguniform("weight_decay", 1e-10, 1e-3), 6)
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)

    # ロガーの設定
    logger: logging.Logger = init_logger(
        config_file=cfg.logger.config_file, name=cfg.logger.name, level=cfg.logger.level
    )
    # trainデータローダーとvalidデータローダーの設定
    data_loader: CustomDataLoader = hydra.utils.instantiate(config=cfg.train_dataloader)
    valid_data_loader = data_loader.split_validation()

    # モデルの定義
    model: nn.Module = hydra.utils.instantiate(
        config=cfg.model,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    # モデル情報の出力
    logger.info(model)

    # モデルをデバイスに転送
    model = model.to(cfg.trainer.device)

    # 複数GPUを使用する場合
    if len(cfg.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.gpu_ids)

    # 損失関数の定義と評価指標の設定
    criterion = hydra.utils.instantiate(config=cfg.loss)

    # 最適化アルゴリズムの設定
    optimizer = torch.optim.Adam(
        model.parameters(), betas=betas, eps=eps, lr=lr, weight_decay=weight_decay
    )

    # トレーナーの設定
    trainer = Trainer(
        model=model,
        logger=logger,
        criterion=criterion,
        optimizer=optimizer,
        epochs=cfg.trainer.epochs,
        early_stop=cfg.trainer.early_stop,
        save_dir=cfg.trainer.save_dir,
        device=cfg.trainer.device,
        train_dataloader=data_loader,
        val_dataloader=valid_data_loader,
    )

    # トレーナーのトレーニングメソッドを呼び出し、検証セットでの最終ロスを返す
    result = trainer.train()

    # 最も良かったエポックの時の検証損失を返す
    return result["monitor_best"]


def objective(trial: FrozenTrial) -> None:
    with initialize(config_path="config/"):
        cfg: Config = compose(config_name="main")

    return main(trial, cfg)


if __name__ == "__main__":
    study: Study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial: FrozenTrial = study.best_trial
    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")
