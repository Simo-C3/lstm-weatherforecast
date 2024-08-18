import os
from logging import INFO, StreamHandler, getLogger

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def export_plot(
    fig: go.Figure, output_path: str, output_name: str, format: str = "png"
):
    match format:
        case "png":
            path = f"{output_path}{output_name}.png"
            fig.write_image(path)  # 画像ファイル出力
            logger.info(f"Saved graph: out: {path}")
        case "html":
            path = f"{output_path}{output_name}.html"
            fig.write_html(path)  # HTMLファイル出力
            logger.info(f"Saved graph: out: {path}")
        case "json":
            path = f"{output_path}{output_name}.json"
            fig.write_json(path)  # JSONファイル出力
            logger.info(f"Saved graph: out: {path}")
        case _:
            raise ValueError(f"Invalid format: {format}")


# 原系列データのプロット
def plot(
    x: pd.Series,
    y: pd.Series,
    output_path: str,
    output_name: str,
    title: str,
    y_label: str = "",
    format: str = "png",
):
    fig = go.Figure()

    # レイアウト設定
    fig.update_layout(
        width=800,
        height=600,
        title=dict(
            text=title,
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
            x=x,
            y=y,
            name=output_name,
            mode="lines",
        )
    )

    # 軸設定
    fig.update_xaxes(
        title_text="年",
        dtick="M12",
        tickformat="%Y",
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
        title_text=y_label,
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

    export_plot(fig, output_path, output_name, format)


# 年別月次平均のプロット
def plot_monthly_average(
    x: pd.Series,
    y: pd.Series,
    output_path: str,
    output_name: str,
    title: str,
    y_label: str = "",
    format: str = "png",
):
    fig = go.Figure()

    # レイアウト設定
    fig.update_layout(
        width=800,
        height=600,
        title=dict(
            text=title,
            xref="paper",
            x=0.5,
            y=0.9,
            xanchor="center",
        ),
        plot_bgcolor="white",
    )

    _df = pd.DataFrame({"date": x, "y": y})

    for year in _df["date"].dt.year.unique():
        _df_year: pd.DataFrame = _df[_df["date"].dt.year == year]
        mean = _df_year.groupby(_df_year["date"].dt.month)["y"].mean()
        fig.add_trace(
            go.Scatter(
                x=mean.index,
                y=mean.values,
                name=str(year),
                mode="lines",
            )
        )

    # 軸設定
    fig.update_xaxes(
        title_text="月",
        dtick="M1",
        tickformat="%m",
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
        title_text=y_label,
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

    export_plot(fig, output_path, output_name, format)


# 自己相関関数のプロット
def plot_autocorr(
    x: pd.Series | np.ndarray,
    y: pd.Series,
    output_path: str,
    output_name: str,
    title: str,
    y_label: str = "",
    K: int = 50,
    R: np.ndarray = None,
    format: str = "png",
) -> None:
    """_summary_

    Args:
        x (pd.Series): _description_
        y (pd.Series): _description_
        output_path (str): _description_
        output_name (str): _description_
        title (str): _description_
        y_label (str, optional): _description_. Defaults to "".
        format (str, optional): _description_. Defaults to "png".

    Raises:
        ValueError: _description_
    """
    fig = make_subplots(
        rows=1,
        cols=5,
        specs=[
            [
                {"colspan": 3, "type": "scatter"},
                None,
                None,
                {"colspan": 2, "type": "bar"},
                None,
            ]
        ],
        subplot_titles=["原系列", "自己相関関数"],
        horizontal_spacing=0.08,
    )

    # レイアウト設定
    fig.update_layout(
        width=1600,
        height=600,
        title=dict(
            text=title,
            xref="paper",
            x=0.5,
            y=0.9,
            xanchor="center",
        ),
        plot_bgcolor="white",
        showlegend=False,
    )

    # オリジナルデータのプロット
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=output_name,
            mode="lines",
        ),
        row=1,
        col=1,
    )

    # 軸設定
    fig.update_xaxes(
        row=1,
        col=1,
        title_text="n",
        dtick=len(x) // 5,
        # グリッドの設定
        gridcolor="lightgrey",
        gridwidth=1,
        # 外枠の設定
        linecolor="black",
        linewidth=1,
        mirror=True,
        # 軸メモリの設定
        ticks="inside",
        tickcolor="gray",
        tickwidth=1,
        ticklen=5,
    )
    fig.update_yaxes(
        row=1,
        col=1,
        title_text=y_label,
        # グリッドの設定
        gridcolor="lightgrey",
        gridwidth=1,
        # 外枠の設定
        linecolor="gray",
        linewidth=1,
        mirror=True,
        # 軸メモリの設定
        ticks="inside",
        tickcolor="black",
        tickwidth=1,
        ticklen=5,
    )

    # 自己相関関数のプロット
    fig.add_trace(
        go.Bar(
            x=list(range(K)),
            y=R,
            name=output_name,
            marker_color="rgba(0, 0, 255, 0.5)",
        ),
        row=1,
        col=4,
    )

    # 軸設定
    fig.update_xaxes(
        row=1,
        col=4,
        title_text="k",
        dtick=K // 2,
        # グリッドの設定
        gridcolor="lightgrey",
        gridwidth=1,
        # 外枠の設定
        linecolor="black",
        linewidth=1,
        mirror=True,
        # 軸メモリの設定
        ticks="inside",
        tickcolor="gray",
        tickwidth=1,
        ticklen=5,
    )
    fig.update_yaxes(
        row=1,
        col=4,
        title_text=r"$R_k$",
        # グリッドの設定
        gridcolor="lightgrey",
        gridwidth=1,
        # 外枠の設定
        linecolor="gray",
        linewidth=1,
        mirror=True,
        # 軸メモリの設定
        ticks="inside",
        tickcolor="black",
        tickwidth=1,
        ticklen=5,
    )

    export_plot(fig, output_path, output_name, format)


# 異常値の処理
def handle_outliers(data: pd.DataFrame) -> None:
    features = ["meantemp", "humidity", "wind_speed", "meanpressure"]
    bounds = {}
    for column in features:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        bounds[column] = (lower_bound, upper_bound)
        data.loc[data[column] < lower_bound, column] = data[column].mean()
        data.loc[data[column] > upper_bound, column] = data[column].mean()


# 自己相関関数の計算
def autocorr(data: pd.Series, k: int) -> np.ndarray:
    """_summary_

    Args:
        data (pd.Series): _description_
        k (int): _description_

    Returns:
        np.ndarray: _description_
    """
    mu = np.mean(data)
    R = np.zeros(k + 1)
    C0 = np.mean(np.sum((data - mu) ** 2))
    for i in range(k + 1):
        Ck = np.mean(np.sum((data - mu) * (data.shift(i) - mu)))
        R[i] = Ck / C0
    return R


if __name__ == "__main__":
    # 定数定義
    FILE_NAME = "DailyDelhiClimateTrain"
    DATA_PATH = "data/"
    OUTPUT_PATH = "outputs/plot/"

    # ロガー設定
    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel(INFO)
    logger.setLevel(INFO)
    logger.addHandler(handler)

    # ディレクトリが存在しない場合は作成
    if os.path.exists(OUTPUT_PATH) is False:
        try:
            os.makedirs(OUTPUT_PATH)
        except OSError as e:
            logger.error(e)

    # データ読み込み
    data = pd.read_csv(f"{DATA_PATH}{FILE_NAME}.csv", header=0)

    # 異常値の処理
    handle_outliers(data)

    # 日付型に変換
    data["date"] = pd.to_datetime(data["date"])

    # 平均気温の原系列データ
    title = "平均気温の原系列データ"
    y_label = r"$y_n$"
    output_name = "MeanTemp"
    plot(data["date"], data["meantemp"], OUTPUT_PATH, output_name, title, y_label)

    # 湿度の原系列データ
    title = "湿度の原系列データ"
    y_label = r"$y_n$"
    output_name = "Humidity"
    plot(data["date"], data["humidity"], OUTPUT_PATH, output_name, title, y_label)

    # 風速の原系列データ
    title = "風速の原系列データ"
    y_label = r"$y_n$"
    output_name = "WindSpeed"
    plot(data["date"], data["wind_speed"], OUTPUT_PATH, output_name, title, y_label)

    # 気圧の原系列データ
    title = "気圧の原系列データ"
    y_label = r"$y_n$"
    output_name = "MeanPressure"
    plot(data["date"], data["meanpressure"], OUTPUT_PATH, output_name, title, y_label)

    # 平均気温の年別月次平均
    title = "平均気温の年別月次平均"
    y_label = r"$y_n$"
    output_name = "MeanTemp-MonthlyAverage"
    plot_monthly_average(
        data["date"], data["meantemp"], OUTPUT_PATH, output_name, title, y_label
    )

    # 湿度の年別月次平均
    title = "湿度の年別月次平均"
    y_label = r"$y_n$"
    output_name = "Humidity-MonthlyAverage"
    plot_monthly_average(
        data["date"], data["humidity"], OUTPUT_PATH, output_name, title, y_label
    )

    # 風速の年別月次平均
    title = "風速の年別月次平均"
    y_label = r"$y_n$"
    output_name = "WindSpeed-MonthlyAverage"
    plot_monthly_average(
        data["date"], data["wind_speed"], OUTPUT_PATH, output_name, title, y_label
    )

    # 気圧の年別月次平均
    title = "気圧の年別月次平均"
    y_label = r"$y_n$"
    output_name = "MeanPressure-MonthlyAverage"
    plot_monthly_average(
        data["date"], data["meanpressure"], OUTPUT_PATH, output_name, title, y_label
    )

    # 平均気温の自己相関関数
    title = "平均気温の自己相関関数"
    y_label = r"$R_k$"
    output_name = "MeanTemp-Autocorr"
    K = 1500
    R = autocorr(data["meantemp"], K)
    plot_autocorr(
        data.index.values,
        data["meantemp"],
        OUTPUT_PATH,
        output_name,
        title,
        y_label,
        K,
        R,
    )

    # 湿度の自己相関関数
    title = "湿度の自己相関関数"
    y_label = r"$R_k$"
    output_name = "Humidity-Autocorr"
    K = 1500
    R = autocorr(data["humidity"], K)
    plot_autocorr(
        data.index.values,
        data["humidity"],
        OUTPUT_PATH,
        output_name,
        title,
        y_label,
        K,
        R,
    )

    # 風速の自己相関関数
    title = "風速の自己相関関数"
    y_label = r"$R_k$"
    output_name = "WindSpeed-Autocorr"
    K = 1500
    R = autocorr(data["wind_speed"], K)
    plot_autocorr(
        data.index.values,
        data["wind_speed"],
        OUTPUT_PATH,
        output_name,
        title,
        y_label,
        K,
        R,
    )

    # 気圧の自己相関関数
    title = "気圧の自己相関関数"
    y_label = r"$R_k$"
    output_name = "MeanPressure-Autocorr"
    K = 1500
    R = autocorr(data["meanpressure"], K)
    plot_autocorr(
        data.index.values,
        data["meanpressure"],
        OUTPUT_PATH,
        output_name,
        title,
        y_label,
        K,
        R,
    )

    # # 平均気温の前年同期比
    # title = "平均気温の前年同期比"
    # y_label = r"$y_n/y_{n-12}$"
    # output_name = "YoY-MeanTemp"
    # data[output_name] = data["meantemp"] / data["meantemp"].shift(
    #     12
    # )  # 12ヶ月前のデータとの比率
    # plot(data["date"], data[output_name], OUTPUT_PATH, output_name, title, y_label)

    # # 湿度の前年同期比
    # title = "湿度の前年同期比"
    # y_label = r"$y_n/y_{n-12}$"
    # output_name = "YoY-Humidity"
    # data[output_name] = data["humidity"] / data["humidity"].shift(
    #     12
    # )  # 12ヶ月前のデータとの比率
    # plot(data["date"], data[output_name], OUTPUT_PATH, output_name, title, y_label)

    # # 風速の前年同期比
    # title = "風速の前年同期比"
    # y_label = r"$y_n/y_{n-12}$"
    # output_name = "YoY-WindSpeed"
    # data[output_name] = data["wind_speed"] / data["wind_speed"].shift(
    #     12
    # )  # 12ヶ月前のデータとの比率
    # plot(data["date"], data[output_name], OUTPUT_PATH, output_name, title, y_label)

    # # 気圧の前年同期比
    # title = "気圧の前年同期比"
    # y_label = r"$y_n/y_{n-12}$"
    # output_name = "YoY-MeanPressure"
    # data[output_name] = data["meanpressure"] / data["meanpressure"].shift(
    #     12
    # )  # 12ヶ月前のデータとの比率
    # plot(data["date"], data[output_name], OUTPUT_PATH, output_name, title, y_label)

    # # 課題1-3　前期比
    # title = "前期比"
    # y_label = r"$y_n/y_{n-1}$"
    # output_name = "QoQ"
    # data["QoQ"] = data["orignal"] / data["orignal"].shift(1)  # 1ヶ月前のデータとの比率
    # plot(data["date"], data["QoQ"], OUTPUT_PATH, output_name, title, y_label)

    # # 課題1-4　対数変換データ
    # title = "対数変換データ"
    # y_label = r"$\log(y_n)$"
    # output_name = "Log"
    # data["Log"] = data["orignal"].apply(lambda x: np.log(x))  # 自然対数
    # plot(data["date"], data["Log"], OUTPUT_PATH, output_name, title, y_label)

    # # 課題1-5　対数変換データの季節階差
    # title = "対数変換データの季節階差"
    # y_label = r"$\log(y_n) - \log(y_{n-12})$"
    # output_name = "LogSeasonDiff"
    # data["LogSeasonDiff"] = data["Log"] - data["Log"].shift(12)
    # plot(data["date"], data["LogSeasonDiff"], OUTPUT_PATH, output_name, title, y_label)

    # # 課題1-6　対数変換データの階差
    # title = "対数変換データの階差"
    # y_label = r"$\log(y_n) - \log(y_{n-1})$"
    # output_name = "LogDiff"
    # data["LogDiff"] = data["Log"] - data["Log"].shift(1)
    # plot(data["date"], data["LogDiff"], OUTPUT_PATH, output_name, title, y_label)
