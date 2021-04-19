# -*- coding: utf-8 -*-

import os
import sys

import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pprint
import matplotlib.pyplot as plt


def load_yaml(path):
    """
    Load YAML file
    """
    _yaml = yaml.safe_load(open(path, "r"))
    return _yaml if _yaml else {}


def error(labels, outputs):
    """
    Calcualte L1 error
    """
    err = F.l1_loss(labels, outputs)
    return err


def save_model(model, optimizer, hparams, appliance, transform, file_name_model, error):
    """
    Save model and metadata to file
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hparams": hparams,
            "appliance": appliance,
            "transform": transform,
            "error": error,
        },
        file_name_model,
    )


def load_model(file_name_model, model, optimizer=None):
    """
    Load model and metadata from file
    """
    if torch.cuda.is_available():
        state = torch.load(file_name_model)
    else:
        state = torch.load(file_name_model, map_location=torch.device("cpu"))

    model.load_state_dict(state["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    hparams = state.get("hparams", None)
    appliance = state.get("appliance", None)

    transform = state.get("transform", None)
    error = state.get("error", None)

    print("=========== ARCHITECTURE ==========")
    print("Reloading appliance")
    pprint.pprint(appliance)
    print("Reloading transform")
    pprint.pprint(transform)
    print("===================================")
    return transform, error


def save_dataset(transform, train_, test_, filename):
    """
    Save training and testing dataset to file
    """
    torch.save({"transform": transform, "train": train_, "test": test_}, filename)


def plot_window(
    x, y, yhat, reg, clas, alphas, loss, err, classification_enabled, filename
):
    """
    Plot sliding window to visualize disaggregation results, keep track
    of results in training or testing and debugging

    Plotting multipel time series
       - Aggregated demand
       - Appliance demand
       - Disaggregation prediction
       - Regression branch prediction
       - Classification branch prediction
    """
    subplt_x = 4
    subplt_y = 4
    plt.figure(1, figsize=(20, 16))
    plt.subplots_adjust(top=0.88)

    idxs = np.random.randint(len(x), size=(subplt_x * subplt_y))
    for i, idx in enumerate(idxs):
        x_, y_, yhat_, reg_, clas_ = (
            x.detach().numpy()[idx][0],
            y.detach().numpy()[idx],
            yhat.detach().numpy()[idx],
            reg.detach().numpy()[idx],
            clas.detach().numpy()[idx],
        )
        alphas_ = alphas.detach().numpy()[idx].flatten()
        ax1 = plt.subplot(subplt_x, subplt_y, i + 1)
        ax2 = ax1.twinx()
        ax1.plot(range(len(x_)), x_, color="b", label="x")
        ax1.plot(range(len(y_)), y_, color="r", label="y")
        ax1.plot(range(len(reg_)), reg_, color="black", label="reg")
        ax1.plot(range(len(yhat_)), yhat_, alpha=0.5, color="orange", label="yhat")
        ax2.fill_between(
            range(len(alphas_)), alphas_, alpha=0.5, color="lightgrey", label="alpha"
        )
        if classification_enabled:
            alphas_max = np.max(alphas_)
            ax2.plot(
                range(len(clas_)),
                clas_ * alphas_max,
                color="cyan",
                alpha=0.25,
                label="reg",
            )

    plt.suptitle(f"loss {loss:.2f} error {err:.2f}")
    ax1.legend()
    ax2.legend()
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
