# -*- coding: utf-8 -*-

import os
import sys
import pprint

from datetime import datetime

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


def summary(path, results):
    """
    Helper method used to save training results
    Plot train vs validation loss and error to diagnose
       - Underfitting
       - Overfitting
       - Good fitting
    """
    df = pd.DataFrame(
        [
            {
                "epoch": x[0][0],
                "train_loss": x[0][1],
                "train_err": x[0][2],
                "eval_loss": x[1][1],
                "eval_err": x[1][2],
            }
            for x in results
        ]
    ).set_index("epoch")

    # Plot train vs eval loss to make diagnose
    columns = ["train_loss", "eval_loss"]
    filename = os.path.join(path, "results-loss.csv")
    df[columns].round(3).to_csv(filename, sep=";")
    filename = os.path.join(path, "results-loss.png")

    plt.figure(1, figsize=(10, 8))
    df[columns].round(3).plot()
    plt.savefig(filename)
    plt.clf()

    # Plot train vs eval error to make diagnose
    columns = ["train_err", "eval_err"]
    filename = os.path.join(path, "results-error.csv")
    df[columns].round(3).to_csv(filename, sep=";")
    filename = os.path.join(path, "results-error.png")

    plt.figure(1, figsize=(10, 8))
    df[columns].round(3).plot()
    plt.savefig(filename)
    plt.clf()


def train_single_epoch(
    epoch, model, train_loader, transform, optimizer, eval_loader, plotfilename=None
):
    """
    Train single epoch for specific model and appliance
    """
    model.train()
    errs, losses = [], []

    start = datetime.now()  # setup a timer for the train
    for idx, (x, y, clas) in enumerate(train_loader):
        # Prepare model input data
        x = torch.unsqueeze(x, dim=1)

        optimizer.zero_grad()
        x, y, clas = x.to(device), y.to(device), clas.to(device)
        yhat, reghat, alphas, clashat = model(x)

        # Calculate prediction loss. See network architecture
        # and loss details in documentation
        loss_out = F.mse_loss(yhat, y)

        # Different loss functions are used depending on model_type
        # If classification is disabled loss function do not take
        # care of classification loss
        if model.classification_enabled:
            loss_clas = F.binary_cross_entropy(clashat, clas)
            loss = loss_out + loss_clas
        else:
            loss = loss_out

        loss.backward()
        optimizer.step()
        err = error(y, yhat)

        loss_, err_ = loss.item(), err.item()
        losses.append(loss_)
        errs.append(err_)

        if idx % 100 == 0:
            # Plotting sliding window samples in order to debug or
            # keep track of current testing process
            print(f"train epoch={epoch} batch={idx+1} loss={loss:.2f} err={err:.2f}")
            if plotfilename:
                filename = plotfilename + f".{idx}.png"
                x = x.cpu()
                y = y.cpu()
                yhat = yhat.cpu()
                reghat = reghat.cpu()
                if transform:
                    # If transform enabled undo standardization in order
                    # to proper visualize regression branch prediction
                    x = (x * transform["sample_std"]) + transform["sample_mean"]
                    y = (y * transform["target_std"]) + transform["target_mean"]
                    yhat = (yhat * transform["target_std"]) + transform["target_mean"]
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
                    loss_,
                    err_,
                    model.classification_enabled,
                    filename,
                )

    end = datetime.now()
    total_seconds = (end - start).seconds
    print("------------------------------------------")
    print(f"Epoch seconds: {total_seconds}")
    print("------------------------------------------")

    return np.mean(losses), np.mean(errs)


def eval_single_epoch(model, eval_loader, transform, plotfilename=None):
    """
    Eval single epoch for specific model and appliance
    """

    errs, losses = [], []
    with torch.no_grad():
        model.eval()
        for idx, (x, y, clas) in enumerate(eval_loader):
            # Prepare model input data
            x = torch.unsqueeze(x, dim=1)

            x, y, clas = x.to(device), y.to(device), clas.to(device)
            yhat, reghat, alphas, clashat = model(x)

            # Calculate prediction loss. See network architecture
            # and loss details in documentation
            loss_out = F.mse_loss(yhat, y)

            # Different loss functions are used depending on model_type
            # If classification is disabled loss function do not take
            # care of classification loss
            if model.classification_enabled:
                loss_clas = F.binary_cross_entropy(clashat, clas)
                loss = loss_out + loss_clas
            else:
                loss = loss_out
            err = error(y, yhat)

            loss_, err_ = loss.item(), err.item()
            losses.append(loss_)
            errs.append(err_)

            if idx % 100 == 0:
                # Plotting sliding window samples in order to debug or
                # keep track of current testing process
                print(f"eval batch={idx+1} loss={loss:.2f} err={err:.2f}")
                if plotfilename:
                    filename = plotfilename + f".{idx}.attention.png"
                    x = x.cpu()
                    y = y.cpu()
                    yhat = yhat.cpu()
                    reghat = reghat.cpu()
                    if transform:
                        # If transform enabled undo standardization in order
                        # to proper visualize regression branch prediction
                        x = (x * transform["sample_std"]) + transform["sample_mean"]
                        y = (y * transform["target_std"]) + transform["target_mean"]
                        yhat = (yhat * transform["target_std"]) + transform[
                            "target_mean"
                        ]
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
                        loss_,
                        err_,
                        model.classification_enabled,
                        filename,
                    )
    return np.mean(losses), np.mean(errs)


def train_model(datapath, output, appliance, hparams, doplot=None, reload=True):
    """
    Train specific model and appliance
    """

    # Load appliance specifications and hyperparameters from
    # settings
    buildings = appliance["buildings"]["train"]
    name = appliance["name"]
    params = appliance["hparams"]
    record_err = np.inf

    # Load whether data transformation is required. See details
    # on data normalization in documentation
    transform_enabled = appliance.get("normalization", False)
    # Load specific network architecture to train
    model_type = appliance.get("model", "ModelPaper")

    # Initialize active settings described in documentation.
    # Used to identify whether an appliance is classified as active
    # Used to enableoversampling to fix sliding windows active/inactive
    # imbalance
    active_threshold = appliance.get("active_threshold", 0.15)
    active_ratio = appliance.get("active_ratio", 0.5)
    active_oversample = appliance.get("active_oversample", 2)

    transform = None  # Data transformation disabled by default

    # Load train dataset
    my_dataset = InMemoryKoreaDataset(
        datapath,
        buildings,
        name,
        windowsize=params["L"],
        active_threshold=active_threshold,
        active_ratio=active_ratio,
        active_oversample=active_oversample,
        transform_enabled=transform_enabled,
    )

    if transform_enabled:
        # Load dataset transformation parameters from dataset
        transform = {
            "sample_mean": my_dataset.sample_mean,
            "sample_std": my_dataset.sample_std,
            "target_mean": my_dataset.target_mean,
            "target_std": my_dataset.target_std,
        }
        print(transform)

    # Size train and evaluation dataset
    total_size = len(my_dataset)
    train_size = int(hparams["train_size"] * (total_size))
    eval_size = total_size - train_size

    print("============= DATASET =============")
    print(f"Total size: {total_size}".format(total_size))
    print(f"Train size: {train_size}".format(train_size))
    print(f"Eval size: {eval_size}".format(eval_size))
    print("===================================")
    print("=========== ARCHITECTURE ==========")
    pprint.pprint(appliance)
    print("===================================")

    # Split and randomize train and evaluation dataset
    train_dataset, eval_dataset = torch.utils.data.random_split(
        my_dataset, (train_size, eval_size)
    )

    # Save train dataset in order to use it in later
    # training sessions or debugging
    filename = os.path.join(output, "dataset.pt")
    save_dataset(transform, train_dataset, eval_dataset, filename)

    # Initialize train dataset loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True
    )
    # Initialize evaluation dataset loader
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=hparams["batch_size"]
    )

    model_type = getattr(nilmmodel, model_type)
    model = model_type(params["L"], params["F"], params["K"], params["H"])
    model = model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), hparams["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    if reload:
        # Reload pretrained model in order to continue
        # previous training sessions
        filename = os.path.join(output, appliance["filename"])
        print("====================================")
        print("Reloading model: ", filename)
        print("====================================")
        transform, record_err = load_model(filename, model, optimizer)

    results = []

    start = datetime.now()
    for epoch in range(hparams["epochs"]):
        # Iterate over training epochs
        filename = os.path.join(output, appliance["filename"] + str(epoch))

        plotfilename = None
        if doplot:
            plotfilename = filename

        err_ = None
        try:
            # Train single epoch
            loss, err = train_single_epoch(
                epoch,
                model,
                train_loader,
                transform,
                optimizer,
                eval_loader,
                plotfilename,
            )
            print("==========================================")
            print(f"train epoch={epoch} loss={loss:.2f} err={err:.2f}")
            print("==========================================")

            loss_, err_ = eval_single_epoch(model, eval_loader, transform)
            print("==========================================")
            print(f"eval loss={loss_:.2f} err={err_:.2f}")
            print("==========================================")

            # tune.report(eval_loss=loss_)
            results.append([(epoch, loss, err), (epoch, loss_, err_)])

            if err_ < record_err:
                # Compare current epoch error against previous
                # epochs error (minimum historic error) to check whether current
                # trained model is better than previous ones (best historic error)
                # Set and save current trained model as best historic trained
                # model if current error is lower than historic error
                filename = os.path.join(output, appliance["filename"])
                save_model(
                    model, optimizer, hparams, appliance, transform, filename, err_
                )
                record_err = err_
        except Exception as e:
            print(e)

        scheduler.step()

    end = datetime.now()
    total_seconds = (end - start).seconds
    print("------------------------------------------")
    print(f"Total seconds: {total_seconds}")
    print("------------------------------------------")

    # Save model training results
    summary(output, results)

    return model, transform


def train_model_wrapper(config):
    """
    Wrapper to adapt model training to tune interface
    """
    datapath = config["datapath"]
    output = config["output"]
    appliance = config["appliance"]
    hparams = config["hparams"]
    doplot = config["doplot"]
    reload = config["reload"]
    tune_hparams = config["tune"]

    appliance["hparams"]["F"] = tune_hparams["F"]
    appliance["hparams"]["K"] = tune_hparams["K"]
    appliance["hparams"]["H"] = tune_hparams["H"]

    return train_model(datapath, output, appliance, hparams, doplot, reload)
