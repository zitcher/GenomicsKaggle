import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # optional progress bar
from transformers import XLNetConfig, XLNetForSequenceClassification

from model import Embeddor
from preprocess import HistoneDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparams = {
    "batch_size": 20,
    "learning_rate": 0.001,
    "num_epochs": 2,
    "embed_size": 264,
    "n_layer": 12,
    "n_head": 6,
    "d_inner": 1024,
}

train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058',
               'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097',
              'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']


def train(model, embeddor, train_loader):
    print("starting train")

    loss_fn = torch.nn.MSELoss(
        size_average=None, reduce=None, reduction='mean')
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
            embedded = embeddor(x)
            y_pred = model(None, inputs_embeds=embedded)[0].squeeze()
            # torch.set_printoptions(profile="full")
            # print("y_pred", y_pred)
            # print("target", y)

            loss = loss_fn(y_pred, y)

            loss.backward()  # calculate gradients
            optimizer.step()  # update model weights

            print("loss:", loss.item())


def validate(model, embeddor, validate_loader):
    print("starting validation")
    loss_fn = torch.nn.MSELoss(
        size_average=None, reduce=None, reduction='mean')

    model = model.eval()
    losses = []

    for batch in tqdm(validate_loader):
        x = batch['x']
        y = batch['y']
        x = x.to(device)
        y = y.to(device)

        embedded = embeddor(x)
        y_pred = model(None, inputs_embeds=embedded)[0].squeeze()

        loss = loss_fn(y_pred, y)
        loss = loss.detach().cpu()

        losses.append(loss.item())

    print("mean loss:", np.mean(losses))


def test(model, embeddor, test_loader):
    print("starting test")
    model = model.eval()
    classification = []

    for batch in tqdm(test_loader):
        x = batch['x']
        cell_type = batch['cell_type']
        id = batch['id']
        x = x.to(device)

        embedded = embeddor(x)
        y_pred = model(None, inputs_embeds=embedded)[0].squeeze()
        y_pred = y_pred.detach().cpu()
        for i in range(y_pred.size()[0]):
            # print(cell_type[i].item(), id[i].item(), y_pred[i].item())
            c_type = cell_type[i].item()
            c_id = str(int(id[i].item()))
            c_label = c_type + "_" + c_id
            score = str(y_pred[i].item())
            classification.append((c_label, score))

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

    model = None
    train_dataset = None
    validate_dataset = None
    test_dataset = None

    embeddor = Embeddor(
        in_size=5, out_size=hyperparams["embed_size"]).to(device)
    configuration = XLNetConfig(
        vocab_size=0,
        num_labels=1,
        d_model=hyperparams["embed_size"],
        n_layer=hyperparams["n_layer"],
        n_head=hyperparams["n_head"],
        d_inner=hyperparams["d_inner"]
    )
    model = XLNetForSequenceClassification(configuration).to(device)

    print("loading data")
    if args.train:
        train_file = args.train[0]
        dataset = HistoneDataset(
            train_file, args.savedata, args.loaddata, "train")

        split_amount = int(len(dataset) * 0.9)
        train_dataset, validate_dataset = random_split(
            dataset, (split_amount, len(dataset) - split_amount))

    if args.test:
        test_file = args.test[0]

        test_dataset = HistoneDataset(
            test_file, args.savedata, args.loaddata, "eval")

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
        test_loader = DataLoader(
            test_dataset, batch_size=hyperparams['batch_size'])

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model, embeddor, train_loader)
        validate(model, embeddor, validate_loader)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
    if args.test:
        print("running testing loop...")
        test(model, embeddor, test_loader)
