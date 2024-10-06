from plotly import graph_objects as go


def plot_loss(df):

    epoch_list = df["epoch"].tolist()
    val_loss_list = df["val_loss"].tolist()
    train_loss_list = df["train_loss"].tolist()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=epoch_list, y=train_loss_list, mode="lines", name="Train Loss"))
    fig.add_trace(go.Scatter(x=epoch_list, y=val_loss_list, mode="lines", name="Validation Loss"))

    fig.update_layout(
        title={"text": "Training/Validation Loss", "x": 0.45, "y": 0.95, "xanchor": "center", "yanchor": "top"},
        xaxis_title="Epoch",
        yaxis_title="Loss",
        xaxis=dict(dtick=1),
        yaxis=dict(range=[0, 2]),
        width=1000,
        height=500,
    )

    fig.show()
