from logging import Logger

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm.contrib.logging import logging_redirect_tqdm

import model


def train(
    model: model.LSTMModel,
    device: any,
    criterion: _Loss,
    optimizer: optim.Optimizer,
    epochs: int,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    logger: Logger,
):
    logger.info("Start training")

    with logging_redirect_tqdm(loggers=[logger]):
        for epoch in range(epochs):
            model.train()
            for train_batch_idx, (data, label) in enumerate(train_dataloader):
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(data)
                train_loss = criterion(output, label)
                train_loss.backward()
                optimizer.step()

            model.eval()

            val_loss = 0
            with torch.no_grad():
                for val_batch_idx, (data, label) in enumerate(valid_dataloader):
                    data, label = data.to(device), label.to(device)
                    output = model(data)
                    loss = criterion(output, label)

                    val_loss += loss.item()

            logger.info(
                f"Epoch: {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss/len(valid_dataloader)}"
            )

    logger.info("Finish training")
