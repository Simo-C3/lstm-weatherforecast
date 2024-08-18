from logging import Logger

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from base import BaseTrainer
from utils import inf_loop


class Trainer(BaseTrainer):
    """
    トレーニングを管理するクラス
    """

    def __init__(
        self,
        model: Module,
        logger: Logger,
        criterion: _Loss,
        optimizer: Optimizer,
        epochs: int,
        early_stop: int,
        save_dir: str,
        device: str,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        lr_scheduler=None,
        len_epoch: int = None,
    ):
        super().__init__(
            model,
            logger,
            criterion,
            optimizer,
            epochs,
            early_stop,
            save_dir,
            device,
            train_dataloader,
            val_dataloader,
        )

        if len_epoch is None:
            # エポック単位でのトレーニング
            self.len_epoch = len(self.train_dataloader)
        else:
            # イテレーション単位でのトレーニング（無限ループ）
            self.train_dataloader = inf_loop(train_dataloader)
            self.len_epoch = len_epoch
        self.do_validation = self.val_dataloader is not None  # 検証を行うかどうか
        self.lr_scheduler = lr_scheduler  # 学習率スケジューラー

    def _train_epoch(self) -> tuple[Module, float]:
        """
        1エポックのトレーニングロジック

        :return: 損失
        """
        self.model.train()  # モデルをトレーニングモードに設定
        total_loss = 0

        for batch_idx, (data, label) in enumerate(self.train_dataloader):
            data: Tensor = data.to(self.device)
            label: Tensor = label.to(self.device)

            self.optimizer.zero_grad()  # 勾配をリセット
            output: Tensor = self.model(data)  # モデルで推論
            loss: Tensor = self.criterion(output, label)  # 損失計算
            loss.backward()  # 勾配の計算
            self.optimizer.step()  # パラメータ更新
            total_loss += loss.item()

            # self.logger.info(self._progress(batch_idx))

        return self.model, total_loss / len(self.train_dataloader)

    def _validate(self) -> float:
        """
        1エポックの検証ロジック

        :return: 検証の情報を含むログ
        """
        self.model.eval()  # モデルを評価モードに設定
        total_loss = 0

        with torch.no_grad():  # 勾配計算を無効化
            for batch_idx, (data, label) in enumerate(self.val_dataloader):
                data: Tensor = data.to(self.device)
                label: Tensor = label.to(self.device)

                output: Tensor = self.model(data)
                loss: Tensor = self.criterion(output, label)
                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def _progress(self, batch_idx):
        """
        進行状況のフォーマット

        :param batch_idx: 現在のバッチインデックス
        :return: 進行状況の文字列
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
