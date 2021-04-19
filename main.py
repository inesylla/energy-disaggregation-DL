# -*- coding: utf-8 -*-
import os
import sys

from datetime import datetime
from argparse import ArgumentParser

import torch
from ray import tune

from utils import error, load_yaml
from train import train_model
from test import test_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_arguments():
    """
    Command line arguments parser
    --settings
      Path to settings yaml file where all disaggregation scenarios
      and model hyperparameters are described
    --appliance
      Name of the appliance to train or test
    --path
      Path to output folder where resuls are saved
    --train
      Set to train or unset to test
    --tune
      Set to enable automatic architecture hyperparameters tunning 
    --epochs
      Number of epochs to train
    --disable-plot
      Disable sliding window plotting during train or test
    --disable-random
      Disable randomness in processing
    """
    parser = ArgumentParser(description="nilm-project")
    parser.add_argument("--settings")
    parser.add_argument("--appliance")
    parser.add_argument("--path")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--epochs")
    parser.add_argument("--disable-plot", action="store_true")
    parser.add_argument("--disable-random", action="store_true")
    return parser.parse_args()


def main():
    """
    Main task called from command line. Command line arguments
    and train or test is launched
    """
    args = get_arguments()

    if args.disable_random:  # Disable randomness
        torch.manual_seed(7)

    train = args.train
    tune_enabled = args.tune
    output = args.path
    plot_disabled = args.disable_plot

    # Load settings from YAML file where generic and appliance
    # specific details and model hyperparmeters are described
    settings = load_yaml(args.settings)
    appliance = args.appliance

    dataset = settings["dataset"]
    hparams = settings["hparams"]
    if args.epochs:
        hparams["epochs"] = int(args.epochs)

    appliance = settings["appliances"][appliance]

    datapath = dataset["path"]
    if train:
        # DO TRAIN

        print("==========================================")
        print(f"Training ONGOING")
        print("==========================================")

        if not tune_enabled:
            # If no automatic hyperparameter tunning is enabled
            # use network hyperparameter from settings and train
            # the model
            model, transform = train_model(
                datapath,
                output,
                appliance,
                hparams,
                doplot=not plot_disabled,
                reload=False,  # Do not reload models by default
            )
        else:
            # If automatic hyperparameter tunning is enabled
            # specify hyperparameters grid search and tune the model
            config = {
                "datapath": datapath,
                "output": output,
                "appliance": appliance,
                "hparams": hparams,
                "doplot": not plot_disabled,
                "reload": False,
                "tune": {
                    "F": tune.grid_search([16, 32, 64]),
                    "K": tune.grid_search([4, 8, 16]),
                    "H": tune.grid_search([256, 512, 1024]),
                },
            }
            analysis = tune.run(
                train_model_wrapper,  # Use wrapper to adapt training model
                metric="val_loss",
                mode="min",
                num_samples=5,
                config=config,
            )
            print("==========================================")
            print(f"Best hyperparameters")
            print(analysis.best_config)
            print("==========================================")

        print("==========================================")
        print(f"Training DONE")
        print("==========================================")
    else:
        # DO TEST

        print("==========================================")
        print(f"Testing ONGOING")
        print("==========================================")
        test_model(datapath, output, appliance, hparams, doplot=not plot_disabled)
        print("==========================================")
        print(f"Testing DONE")
        print("==========================================")


if __name__ == "__main__":
    main()
