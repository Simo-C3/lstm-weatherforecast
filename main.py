import logging
from test import test

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.nn.modules.loss import _Loss

from config import Config
from dataloader import CustomDataLoader
from trainer import Trainer
from utils import init_logger

CONFIG_PATH = "config/config.yaml"


def init_seed(seed: int) -> None:
    torch.manual_seed(seed)  # PyTorchの乱数シードを固定
    torch.backends.cudnn.deterministic = True  # 計算の再現性を保証
    torch.backends.cudnn.benchmark = False  # 性能向上のための最適化を無効化
    np.random.seed(seed)  # NumPyのランダムシードを固定


def init_train(
    cfg: Config,
    logger: logging.Logger,
) -> tuple[nn.Module, optim.Optimizer, _Loss, Trainer]:
    model: nn.Module = hydra.utils.instantiate(config=cfg.model)
    train_dataloader: CustomDataLoader = hydra.utils.instantiate(
        config=cfg.train_dataloader
    )
    optimizer: optim.Optimizer = hydra.utils.instantiate(
        config=cfg.optimizer,
        params=model.parameters(),
    )
    criterion: _Loss = hydra.utils.instantiate(config=cfg.loss)
    trainer: Trainer = hydra.utils.instantiate(
        config=cfg.trainer,
        model=model,
        logger=logger,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=train_dataloader.split_validation(),
    )

    return model, optimizer, criterion, trainer


def init_test(cfg: Config) -> tuple[nn.Module, _Loss, CustomDataLoader]:
    model: nn.Module = hydra.utils.instantiate(config=cfg.model)
    test_dataloader: CustomDataLoader = hydra.utils.instantiate(
        config=cfg.test_dataloader
    )
    criterion: _Loss = hydra.utils.instantiate(config=cfg.loss)

    return model, criterion, test_dataloader


@hydra.main(version_base=None, config_path="config/", config_name="main")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))

    init_seed(cfg.seed)

    logger: logging.Logger = init_logger(
        config_file=cfg.logger.config_file, name=cfg.logger.name, level=cfg.logger.level
    )

    if cfg.target == "test":
        model, criterion, test_dataloader = init_test(cfg)
        test(
            model=model,
            model_path=f"{cfg.test.save_dir}/best.pth",
            device=cfg.test.device,
            criterion=criterion,
            test_dataloader=test_dataloader,
            logger=logger,
        )
        return
    else:
        model, _, criterion, trainer = init_train(cfg, logger)

        logger.info("Start training")
        result = trainer.train()
        if result is None:
            logger.error(f"Training failed")
            return
        logger.info("Training finished")

        logger.info(f"Best validation loss: {result['monitor_best']}")
        logger.info(f"Best model epoch: {result['epoch']}")

        model, criterion, test_dataloader = init_test(cfg)
        test(
            model=model,
            model_path=f"{cfg.test.save_dir}/best.pth",
            device=cfg.test.device,
            criterion=criterion,
            test_dataloader=test_dataloader,
            logger=logger,
        )


if __name__ == "__main__":
    main()
