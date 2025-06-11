import os
import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from flask import send_from_directory

import pandas as pd
import plotly.express as px

csv_path = "data/voice_cluster.csv"
audio_base_path = "/run/media/chr1s/0c135d7a-de30-4062-a5bc-16addf703a18/uni/SAudio/datasets/CV/en/clips/"

data = pd.read_csv(csv_path, index_col="clip_id")

data = data.drop(columns=["voice_age", "features_extracted"])

ignore_columns = ["x", "y", "z", "voice_name", "audio_file_name"]
color_columns = [col for col in data.columns if col not in ignore_columns]


def file_name_to_path(file_name):
    return os.path.join(audio_base_path, file_name)


def file_name_to_url(file_name):
    return f"/audio/{file_name}"


# --- Dash app setup ---
app = Dash(__name__)
server = app.server


# Flask route to serve audio files
@server.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(audio_base_path, filename)


# 3D scatter plot with hover info
fig = px.scatter_3d(
    data,
    x="x",
    y="y",
    z="z",
    color=color_columns[0] if color_columns else None,
    hover_data={"voice_name": True, "voice_age_group": True, "audio_file_name": False},
    custom_data=["audio_file_name"],
)
fig.update_traces(marker=dict(size=5))

app.layout = html.Div(
    [
        dcc.Dropdown(
            id="color-dropdown",
            options=[{"label": col, "value": col} for col in color_columns],
            value=color_columns[0] if color_columns else None,
            clearable=False,
            style={"width": "300px", "marginBottom": "20px"},
        ),
        dcc.Graph(
            id="voice-3d-plot",
            figure=fig,
            style={"height": "80vh"},
        ),
        html.Audio(
            id="audio-player",
            src="",
            controls=True,
            autoPlay=True,
            style={"width": "100%", "marginTop": "20px"},
        ),
    ]
)


# Callback: play audio when point is clicked
@app.callback(
    Output("audio-player", "src"),
    Input("voice-3d-plot", "clickData"),
    prevent_initial_call=True,
)
def play_audio(clickData):
    if clickData and "points" in clickData and clickData["points"]:
        file_name = clickData["points"][0]["customdata"][0]
        return file_name_to_url(file_name)
    return ""


# Callback: update 3D plot color by dropdown selection
@app.callback(
    Output("voice-3d-plot", "figure"),
    Input("color-dropdown", "value"),
)
def updateGraph(selected_color):
    fig = px.scatter_3d(
        data,
        x="x",
        y="y",
        z="z",
        color=selected_color,
        hover_data={
            "voice_name": True,
            "voice_age_group": True,
            "audio_file_name": False,
        },
        custom_data=["audio_file_name"],
    )
    fig.update_traces(marker=dict(size=3, opacity=0.9))  # full color
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050, host="0.0.0.0")
