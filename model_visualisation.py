import plotly.express as px


def model_performance(model):
    # plot the loss and val_loss
    fig = px.line(model.history[["loss", "val_loss"]], labels=["Loss", "Val_Loss"])
    fig.update_xaxes(title="Epochs")
    fig.update_yaxes(title="Value")
    fig.show()
