import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # optional progress bar

from model import densenet
from preprocess import HistoneDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparams = {
    "num_epochs": 20,
    "batch_size": 100,
    "learning_rate": 0.001,
}


def cells_to_id(cells):
    cti = dict()
    for i, cell in enumerate(cells):
        cti[cell] = i
    return cti

train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058',
               'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

train_cells_dict = cells_to_id(train_cells)

eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097',
              'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

eval_cells_dict = cells_to_id(eval_cells)


def train(model, train_loader):
    print("starting train")

    loss_fn = torch.nn.MSELoss(
        size_average=None, reduce=None, reduction='mean')
    optimizer = optim.Adam(model.parameters(), hyperparams['learning_rate'])

    model = model.train()
    losses = []
    for epoch in range(hyperparams['num_epochs']):
        for batch in tqdm(train_loader):
            x = batch['x']
            y = batch['y']
            x = x.unsqueeze(1)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)

            loss = loss_fn(y_pred.squeeze(1), y)

            loss.backward()  # calculate gradients
            optimizer.step()  # update model weights

            losses.insert(0, loss.item())
            losses = losses[:100]
            print("loss:", loss.item())
            print("avg loss:", np.mean(losses))


def validate(model, validate_loader):
    print("starting validation")
    loss_fn = torch.nn.MSELoss(
        size_average=None, reduce=None, reduction='mean')

    model = model.eval()
    losses = []

    for batch in tqdm(validate_loader):
        x = batch['x']
        y = batch['y']
        x = x.unsqueeze(1)
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)

        loss = loss_fn(y_pred.squeeze(1), y)

        losses.append(loss.item())

    print("mean loss:", np.mean(losses))


def test(models, test_loaders, cell_types):
    print("starting test")
    classification = []
    for cell in cell_types:
        test_loader = test_loaders[cell]
        cell_models = []
        if cell in train_cells:
            model = models[train_cells_dict[cell]].to(device)
            model = model.eval()
            cell_models.append(model)
        else:
            for cell in train_cells:
                model = models[train_cells_dict[cell]].to(device)
                model = model.eval()
                cell_models.append(model)

        for batch in tqdm(test_loader):
            x = batch['x']
            id = batch['id']
            cell_type = batch['cell_type']
            x = x.unsqueeze(1)
            x = x.to(device)

            y_preds = None
            if len(cell_models) == 1:
                y_preds = cell_models[0](x).unsqueeze(0)
            else:
                for model in cell_models:
                    if y_preds is None:
                        y_preds = model(x).unsqueeze(0)
                    else:
                        y_preds = torch.cat((y_preds, model(x).unsqueeze(0)))

            y_pred = torch.mean(y_preds, 0)
            # print(y_pred)

            for i in range(y_pred.size()[0]):
                # print(cell_type[i].item(), id[i].item(), y_pred[i].item())
                classification.append(
                    (cell_type[i].item() + "_" + str(int(id[i].item())), str(y_pred[i].item())))

    df = pd.DataFrame(classification, columns=['id', 'expression'])
    df.to_csv('submission.csv', index=False)


# python histone.py -s -T data/train.npz -t data/eval.npz
# python histone.py -l -T data/train.npz -t data/eval.npz
# python histone.py -l -t data/eval.npz
# mv model.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", nargs=1, help="train model")
    parser.add_argument("-t", "--test", nargs=1, help="test model")
    args = parser.parse_args()

    print("Device", device)

    models = dict()
    for cell in train_cells:
        model = densenet()
        models[train_cells_dict[cell]] = model

    train_datasets = dict()
    validate_datasets = dict()
    test_datasets = dict()

    if args.train:
        train_file = args.train[0]
        for cell in train_cells:
            dataset = HistoneDataset(train_file, cell)

            split_amount = int(len(dataset) * 0.95)

            train_dataset, validate_dataset = random_split(
                dataset, (split_amount, len(dataset) - split_amount))

            train_datasets[train_cells_dict[cell]] = train_dataset
            validate_datasets[train_cells_dict[cell]] = validate_dataset


    test_file = None
    cell_types = None
    if args.test:
        test_file = args.test[0]
        npzfile = np.load(test_file)
        cell_types = npzfile.files
        for cell in cell_types:
            assert cell in eval_cells_dict
            test_dataset = HistoneDataset(test_file, cell)
            test_datasets[cell] = test_dataset

    train_loaders = dict()
    validate_loaders = dict()
    if args.train:
        for cell in train_cells:
            train_loader = DataLoader(
                train_datasets[train_cells_dict[cell]], batch_size=hyperparams['batch_size'], shuffle=True
            )
            validate_loader = DataLoader(
                validate_datasets[train_cells_dict[cell]], batch_size=hyperparams['batch_size'], shuffle=True
            )

            train_loaders[train_cells_dict[cell]] = train_loader
            validate_loaders[train_cells_dict[cell]] = validate_loader

    test_loaders = dict()
    if args.test:
        for cell in cell_types:
            test_loader = DataLoader(test_datasets[cell], batch_size=hyperparams['batch_size'])
            test_loaders[cell] = test_loader

    if args.load:
        print("loading saved model...")
        for cell in train_cells:
            models[train_cells_dict[cell]].load_state_dict(torch.load('./model' + cell + '.pt'))
    if args.train:
        for i, cell in enumerate(train_cells):
            print("running training loop", i, "out of", len(train_cells), "for", cell)
            model = models[train_cells_dict[cell]].to(device)
            train(model, train_loaders[train_cells_dict[cell]])
            validate(model, validate_loaders[train_cells_dict[cell]])
    if args.save:
        print("saving model...")
        for cell in train_cells:
            torch.save(models[train_cells_dict[cell]].state_dict(), './model'+ cell + '.pt')
    if args.test:
        print("running testing loop...")
        test(models, test_loaders, cell_types)
