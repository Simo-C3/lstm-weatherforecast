import os
from abc import abstractmethod
from logging import Logger

import torch
from numpy import inf
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class BaseTrainer:
    """
    すべてのトレーナーの基底クラス
    """

    logger: Logger
    model: Module
    criterion: _Loss
    metric_ftns: list
    optimizer: Optimizer
    train_dataloader: DataLoader
    val_dataloader: DataLoader

    def __init__(
        self,
        model: Module,
        logger: Logger,
        criterion: _Loss,
        optimizer: Optimizer,
        epochs: int,
        early_stop: int,
        save_dir: str,
        device: any,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
    ):
        self.logger = logger  # ロガー
        self.model = model  # モデル

        self.criterion = criterion  # 損失関数
        self.optimizer = optimizer  # 最適化アルゴリズム

        self.epochs = epochs  # 総エポック数
        self.early_stop = early_stop  # 早期終了の閾値
        self.device = device  # デバイス

        # self.save_period = cfg_trainer["save_period"]  # 保存間隔
        # self.monitor = cfg_trainer.get("monitor", "off")  # モデルの監視設定

        # # モデル性能の監視とベストモデルの保存の設定
        # if self.monitor == "off":
        #     self.mnt_mode = "off"
        #     self.mnt_best = 0
        # else:
        #     self.mnt_mode, self.mnt_metric = self.monitor.split()
        #     assert self.mnt_mode in ["min", "max"]  # 監視モードは'min'または'max'

        #     self.mnt_best = (
        #         inf if self.mnt_mode == "min" else -inf
        #     )  # 監視する最良の値を初期設定
        #     self.early_stop = cfg_trainer.get("early_stop", inf)  # 早期終了の閾値
        #     if self.early_stop <= 0:
        #         self.early_stop = inf  # 早期終了が設定されていない場合

        self.start_epoch = 1  # 開始エポック

        self.checkpoint_dir = save_dir  # チェックポイントの保存ディレクトリ

        # if config.resume is not None:
        #     self._resume_checkpoint(config.resume)  # チェックポイントからの再開

        self.train_dataloader = train_dataloader  # トレーニングデータローダー
        self.val_dataloader = val_dataloader  # 検証データローダー
        self.best_val = 100000  # 検証前の損失

        self.model.to(self.device)  # デバイスへのモデルの配置

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    @abstractmethod
    def _train_epoch(self) -> tuple[Module, float]:
        """
        エポックごとのトレーニングロジック

        :return: モデル, 損失
        """
        raise NotImplementedError  # サブクラスで実装が必要

    @abstractmethod
    def _validate(self) -> float:
        """
        エポックごとの検証ロジック

        :return: 損失
        """
        raise NotImplementedError

    def train(self):
        """
        トレーニングの全体的なロジック
        """
        not_improved_count = 0  # 改善されなかった回数のカウント
        best_state: dict | None = None  # 最良の状態

        for epoch in range(self.start_epoch, self.epochs + 1):
            # エポックごとのトレーニングを実行
            model, train_loss = self._train_epoch()
            # エポックごとの検証実行
            val_loss = self._validate()

            # ログに出力
            self.logger.info(
                f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if self.best_val < val_loss:
                not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break
            else:
                not_improved_count = 0
                self.best_val = val_loss

                arch = type(self.model).__name__
                best_state = {
                    "arch": arch,  # モデルのアーキテクチャ名
                    "epoch": epoch,  # 現在のエポック数
                    "model_state_dict": model.state_dict(),  # モデルの状態
                    "optimizer_state_dict": self.optimizer.state_dict(),  # オプティマイザの状態
                    "monitor_best": self.best_val,  # 監視している最良の評価値
                    # "config": self.config,  # トレーニングの設定
                }
                torch.save(best_state, f"{self.checkpoint_dir}/best.pth")

        return best_state
