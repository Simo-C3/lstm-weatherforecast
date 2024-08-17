from itertools import repeat


def inf_loop(data_loader):
    """
    データローダーを無限に繰り返すラッパー関数
    """
    for loader in repeat(data_loader):
        yield from loader  # データローダーからデータを無限に生成
