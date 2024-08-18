from logging import Logger

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm.contrib.logging import logging_redirect_tqdm

import model
from utils.fig_export import export_plot


def plot(
    dataset: any,
    total_output: list[float],
    output_path: str = "outputs/",
    output_name: str = "result",
):

    fig = go.Figure()

    # レイアウト設定
    fig.update_layout(
        width=800,
        height=600,
        title=dict(
            text="予測結果",
            xref="paper",
            x=0.5,
            y=0.9,
            xanchor="center",
        ),
        plot_bgcolor="white",
    )

    # オリジナルデータのプロット
    fig.add_trace(
        go.Scatter(
            x=dataset["date"],
            y=dataset["meantemp"],
            name="original",
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataset["date"],
            y=total_output,
            name="predicted",
            mode="lines",
        )
    )

    # 軸設定
    fig.update_xaxes(
        title_text="年",
        dtick="M3",
        tickformat="%Y/%m",
        # グリッドの設定
        gridcolor="lightgrey",
        gridwidth=1,
        # 外枠の設定
        linecolor="black",
        linewidth=1,
        mirror=True,
        # 軸メモリの設定
        ticks="inside",
        tickcolor="black",
        tickwidth=1,
        ticklen=5,
    )
    fig.update_yaxes(
        title_text="気温",
        # グリッドの設定
        gridcolor="lightgrey",
        gridwidth=1,
        # 外枠の設定
        linecolor="black",
        linewidth=1,
        mirror=True,
        # 軸メモリの設定
        ticks="inside",
        tickcolor="black",
        tickwidth=1,
        ticklen=5,
    )

    export_plot(fig, output_path, output_name, "png")


def test(
    model: nn.Module,
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
    total_output = np.array([])

    with torch.no_grad():
        with logging_redirect_tqdm(loggers=[logger]):
            for val_batch_idx, (data, label) in enumerate(test_dataloader):
                data: Tensor = data.to(device)
                label: Tensor = label.to(device)

                output: Tensor = model(data)
                total_output = np.append(
                    total_output, torch.flatten(output.cpu()).numpy()
                )
                loss: Tensor = criterion(output, label)
                test_loss += loss.item()

    logger.info(f"Test Loss: {test_loss / len(test_dataloader):.6f} ")

    logger.info(f"Total Output len: {len(total_output)}")
    logger.info(
        f"test_dataloader.dataset.data len: {len(test_dataloader.dataset.data)}"
    )

    plot(test_dataloader.dataset.data, total_output)
