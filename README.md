# 天気予報士 AI

本レポジトリは，過去30日間の平均気温，湿度，風速，気圧のデータから次の日の天気を予測する機械学習モデルを構築するものです．

## Model

LSTM

## Usage

### train

以下のコマンドを実行し，学習を実行してください．

```bash
pipenv run python main.py
```

### パラメータチューニング

ベイズ最適化を用いたパラメータチューニングは以下のコマンドを実行し，学習してください．

```bash
pipenv run python paramtune.py
```
