from preprocess import HistoneDataset
from model import CNN
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, hyperparams):
    pass

def test(model, test_loader, hyperparams):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true", nargs="1",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true", nargs="1",
                        help="run testing loop")
    parser.add_argument("-m", "--multilingual-tags", nargs="*", default=[None],
                        help="target tags for translation")
    args = parser.parse_args()


    hyperparams = {
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "num_kernels": 3,
        "kernel_size": (4, 20),
        "output_size": 2,
        "pool_size": (4, 4),
        "num_epochs": 1,
        "batch_size": 50,
        "learning_rate": 0.001
    }

    model = None
    train_dataset = None
    test_dataset = None
    if args.train:
        train_file = args.train[0]
        train_dataset = HistoneDataset(train_file)
        model = CNN(
            channels=1,
            width=train_dataset.width,
            height=train_dataset.height,
            stride=hyperparams["stride"],
            padding=hyperparams["padding"],
            dilation=hyperparams["dilation"],
            groups=hyperparams["groups"],
            num_kernels=hyperparams["num_kernels"],
            kernel_size=hyperparams["kernel_size"],
            output_size=hyperparams["output_size"],
            pool_size=hyperparams["pool_size"]
        ).to(device)

    if args.test:
        test_file = args.test[0]
        test_dataset = HistoneDataset(test_file)
        if model is None:
            model = CNN(
                channels=1,
                width=train_dataset.width,
                height=train_dataset.height,
                stride=hyperparams["stride"],
                padding=hyperparams["padding"],
                dilation=hyperparams["dilation"],
                groups=hyperparams["groups"],
                num_kernels=hyperparams["num_kernels"],
                kernel_size=hyperparams["kernel_size"],
                output_size=hyperparams["output_size"],
                pool_size=hyperparams["pool_size"]
            ).to(device)

    if args.train:
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparams['batch_size'], shuffle=True
        )

    test_loader = None
    if args.test:
        test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model, train_loader, hyperparams)
    if args.test:
        print("running testing loop...")
        test(model, test_loader, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
