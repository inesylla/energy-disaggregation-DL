# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model as nilmmodel
import matplotlib.pyplot as plt

from dataset import InMemoryKoreaDataset
from utils import error
from utils import save_model, load_model, save_dataset
from utils import plot_window

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_single(
    model, test_loader, transform, appliance, batch_size=64, plotfilename=None
):
    """
    Test specific pretrained model and appliance on test dataset 
    """

    errs, losses = [], []

    L = appliance["hparams"]["L"]
    window_index = np.array(range(L))

    # The disaggregation phase, also carried out with a sliding window
    # over the aggregated signal with hop size equal to 1 sample,
    # generates overlapped windows of the disaggregated signal.
    # reconstruct the overlapped windows by an means of a median
    # filter on the overlapped portions.

    # Use buffer to register overlapped result and apply median filter
    overlapped_y = {}
    overlapped_yhat = {}

    with torch.no_grad():
        model.eval()
        for idx, (x, y, clas) in enumerate(test_loader):
            # model input data
            x = torch.unsqueeze(x, dim=1)

            x, y, clas = x.to(device), y.to(device), clas.to(device)
            yhat, reghat, alphas, clashat = model(x)

            # Force loss to 0 in test in order to reuse current implementation
            # but not used in testing analysis
            loss = 0.0

            # Calculate and use error to evaluate prediction
            err = error(y, yhat)

            err_ = err.item()
            losses.append(loss)
            errs.append(err_)

            x = x.cpu()
            y = y.cpu()
            yhat = yhat.cpu()
            if transform:
                # If transform enabled undo standardization in order to
                # prope evaluate error (paper benchmarking) and visualization
                if (
                    transform["sample_mean"]
                    and transform["sample_std"]
                    and transform["target_mean"]
                    and transform["target_std"]
                ):

                    # Undo standarization
                    x = (x * transform["sample_std"]) + transform["sample_mean"]
                    y = (y * transform["target_std"]) + transform["target_mean"]
                    yhat = (yhat * transform["target_std"]) + transform["target_mean"]

            if idx % 100 == 0:
                # Plotting sliding window samples in order to debug or
                # keep track of current testing process
                print(f"test batch={idx+1} loss={loss:.2f} err={err:.2f}")
                if plotfilename:
                    filename = plotfilename + f".{idx}.attention.png"
                    reghat = reghat.cpu()
                    if transform:
                        # If transform enabled undo standardization in order
                        # to proper visualize regression branch prediction
                        if transform["target_std"] and transform["target_mean"]:
                            reghat = (reghat * transform["sample_std"]) + transform[
                                "sample_mean"
                            ]
                            # Tricky workaround to rescale regression output and make
                            # it easier to visualize and interpret results
                            reghat = reghat / 10.0
                    plot_window(
                        x,
                        y,
                        yhat,
                        reghat,
                        clashat.cpu(),
                        alphas.cpu(),
                        loss,
                        err_,
                        model.classification_enabled,
                        filename,
                    )

            y = y.numpy()
            yhat = yhat.numpy()

            # Update overlapping windows buffer to calculate median filter
            for offset, yy, yyhat in zip(range(batch_size), y, yhat):
                index = (idx * batch_size) + window_index + offset

                for index_, yy_, yyhat_ in zip(index, yy, yyhat):
                    overlapped_y[index_] = yy_
                    overlapped_yhat.setdefault(index_, [])
                    overlapped_yhat[index_].append(yyhat_)

                    if len(overlapped_yhat[index_]) == L:
                        # Calculate median if all overlapped windows in specfic
                        # index are already available. Done prevent memory
                        # overrun
                        overlapped_yhat[index_] = np.median(
                            np.array(overlapped_yhat[index_])
                        )
    # Final buffers with sigle-point single-prediction after median filter
    final_y = []
    final_yhat = []
    index = sorted(list(overlapped_yhat.keys()))

    # Calculate median if all overlapped windows in specfic
    # index are already available. Done prevent memory
    # overrun
    for i in index:
        if isinstance(overlapped_yhat[i], list):
            overlapped_yhat[i] = np.median(np.array(overlapped_yhat[i]))

        # Update final prediction buffers
        final_yhat.append(overlapped_yhat[i])
        final_y.append(overlapped_y[i])

    final_y = np.array(final_y)
    final_yhat = np.array(final_yhat)

    filename = plotfilename + f".result.csv"
    result = pd.DataFrame({"y": final_y, "yhat": final_yhat})
    result.to_csv(filename, index=None, sep=";")

    # Calculate MAE over single-point single-prediction time series
    return np.nanmean(np.abs(final_yhat - final_y))


def test_model(datapath, output, appliance, hparams, doplot=None):
    """
    Test specific pretrained model and appliance on testing
    dataset
    """

    # Load appliance specifications and model hyperparameters
    # from settings

    buildings = appliance["buildings"]["test"]
    name = appliance["name"]

    batch_size = hparams["batch_size"]
    params = appliance["hparams"]

    transform_enabled = appliance.get("normalization", False)
    model_type = appliance.get("model", "ModelPaper")

    # Initialize model network architecture using specified
    # hyperaparameters in settings
    model_type = getattr(nilmmodel, model_type)
    model = model_type(params["L"], params["F"], params["K"], params["H"])
    model = model.to(device)

    # Load pretrained mofrl from file
    name = appliance["name"]
    filename = os.path.join(output, appliance["filename"])
    transform, record_err = load_model(filename, model)

    if not transform_enabled:
        transform = None

    filename = os.path.join(output, appliance["filename"])
    plotfilename = None
    if doplot:
        plotfilename = filename

    # Initialize active settings described in documentation.
    # Used to identify whether an appliance is classified as active
    # Used to enableoversampling to fix sliding windows active/inactive
    # imbalance
    active_threshold = appliance.get("active_threshold", 0.15)
    active_ratio = appliance.get("active_ratio", 0.5)
    active_oversample = appliance.get("active_oversample", 2)

    # Load test dataset
    my_dataset = InMemoryKoreaDataset(
        datapath,
        buildings,
        name,
        windowsize=params["L"],
        active_threshold=False,
        active_ratio=False,
        active_oversample=False,
        transform_enabled=transform_enabled,
        transform=None,  # Using test standarization
        # NOTE: Enable this to use training standarization
        # transform=transform,
    )

    # Load dataset transformation parameters from training
    transform = {
        "sample_mean": my_dataset.sample_mean,
        "sample_std": my_dataset.sample_std,
        "target_mean": my_dataset.target_mean,
        "target_std": my_dataset.target_std,
    }

    # Initialized test data loader using settings batch size
    test_loader = torch.utils.data.DataLoader(
        my_dataset, batch_size=hparams["batch_size"]
    )

    # Launch testing on test dataset
    output = os.path.join(output, f"{name}")
    err = test_single(
        model, test_loader, transform, appliance, batch_size, plotfilename
    )
    print(f"Test err={err:.2f}")
