from preprocess import HistoneDataset
from model import Linear
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, hyperparams):
    print("starting train")

    loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    optimizer = optim.Adam(model.parameters(), hyperparams['learning_rate'])

    model = model.train()
    for epoch in range(hyperparams['num_epochs']):
        for batch in tqdm(train_loader):
            x = batch['x']
            y = batch['y']
            # print("x", x.size())
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)

            loss = loss_fn(y_pred, y)

            loss.backward()  # calculate gradients
            optimizer.step()  # update model weights

            print("loss:", loss.item())


def validate(model, validate_loader, hyperparams):
    print("starting validation")
    loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    model = model.eval()
    losses = []

    for batch in tqdm(validate_loader):
        x = batch['x']
        y = batch['y']
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)

        losses.append(loss.item())

    print("mean loss:", np.mean(losses))


def test(model, test_loader, hyperparams):
    print("starting test")
    model = model.eval()
    classification = []

    for batch in tqdm(test_loader):
        x = batch['x']
        cell_type = batch['cell_type']
        id = batch['id']
        x = x.to(device)

        y_pred = model(x)
        for i in range(y_pred.size()[0]):
            # print(cell_type[i].item(), id[i].item(), y_pred[i].item())
            classification.append((cell_type[i].item() + "_" + str(int(id[i].item())), str(y_pred[i].item())))

    df = pd.DataFrame(classification, columns=['id', 'expression'])
    df.to_csv('submission.csv', index=False)

# python histone.py -s -S ./data -T data/train.npz -t data/eval.npz
# python histone.py -s -L ./data -T data/train.npz -t data/eval.npz
# python histone.py -lL ./data -t data/eval.npz
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", nargs=1, help="train model")
    parser.add_argument("-t", "--test", nargs=1, help="test model")
    parser.add_argument("-S", "--savedata", nargs=1, help="save data")
    parser.add_argument("-L", "--loaddata", nargs=1, help="load data")
    args = parser.parse_args()

    hyperparams = {
        "batch_size": 50,
        "learning_rate": 0.001,
        "num_epochs": 2
    }

    model = None
    train_dataset = None
    validate_dataset = None
    test_dataset = None

    if args.train:
        train_file = args.train[0]
        dataset = HistoneDataset(train_file, args.savedata, args.loaddata, "train")

        split_amount = int(len(dataset) * 0.9)

        model = Linear().to(device)

        train_dataset, validate_dataset = random_split(
            dataset, (split_amount, len(dataset) - split_amount))

    if args.test:
        test_file = args.test[0]

        test_dataset = HistoneDataset(test_file, args.savedata, args.loaddata, "eval")

        if model is None:
            model = Linear().to(device)

    train_loader = None
    if args.train:
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparams['batch_size'], shuffle=True
        )
        validate_loader = DataLoader(
            validate_dataset, batch_size=hyperparams['batch_size'], shuffle=True
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
        validate(model, validate_loader, hyperparams)
    if args.test:
        print("running testing loop...")
        test(model, test_loader, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
