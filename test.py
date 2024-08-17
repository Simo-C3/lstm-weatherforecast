from logging import Logger

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import f1_score
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm.contrib.logging import logging_redirect_tqdm

import model


def test(
    model: model.LSTMModel,
    model_path: str,
    device: any,
    criterion: _Loss,
    test_dataloader: DataLoader,
    logger: Logger,
):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_loss = 0
    with torch.no_grad():
        with logging_redirect_tqdm(loggers=[logger]):
            for val_batch_idx, (data, label) in enumerate(test_dataloader):
                data: Tensor = data.to(device)
                label: Tensor = label.to(device)

                output = model(data)
                loss = criterion(output, label)
                test_loss += loss.item()

    logger.info(f"Test Loss: {test_loss / len(test_dataloader):.6f} ")
