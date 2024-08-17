from abc import abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Variable


class BaseModel(nn.Module):
    """
    ベースモデルクラス
    """

    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播"""
        raise NotImplementedError


class LSTMModel(BaseModel):
    """
    LSTMを使用したモデル

    Attributes
    ----------
    batch_size : int
        バッチサイズ
    input_size : int
        入力次元数
    output_size : int
        出力次元数
    hidden_size : int
        隠れ層の次元数
    num_layers: int
        LSTMの層数
    dropout: float
        ドロップアウト率
    lstm : nn.LSTM
        LSTM層
    fc : nn.Linear
        全結合層
    """

    batch_size: int
    input_size: int
    output_size: int
    hidden_size: int
    num_layers: int
    dropout: float

    lstm: nn.LSTM
    fc: nn.Linear

    def __init__(
        self,
        batch_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the LSTM model.

        Args:
            batch_size (int): バッチサイズ
            input_size (int): 入力次元数
            hidden_size (int): 隠れ層の次元数
            output_size (int): 出力次元数
            num_layers (int): LSTMの層数
            dropout (float): ドロップアウト率
        """
        super(LSTMModel, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # バッチサイズ，シーケンス長，特徴量の次元を取得
        B, T, N = x.size()

        # LSTMの初期状態を定義
        h_0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)

        # LSTMの順伝播
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        # LSTMの出力を全結合層に入力
        out = self.fc(out[:, -1, :])

        return out
