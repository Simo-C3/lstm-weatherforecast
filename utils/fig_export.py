import plotly.graph_objects as go


def export_plot(
    fig: go.Figure, output_path: str, output_name: str, format: str = "png"
):
    match format:
        case "png":
            path = f"{output_path}{output_name}.png"
            fig.write_image(path)  # 画像ファイル出力
        case "html":
            path = f"{output_path}{output_name}.html"
            fig.write_html(path)  # HTMLファイル出力
        case "json":
            path = f"{output_path}{output_name}.json"
            fig.write_json(path)  # JSONファイル出力
        case _:
            raise ValueError(f"Invalid format: {format}")
