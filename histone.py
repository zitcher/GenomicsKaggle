from preprocess import HistoneDataset
from model import densenet
from torch.utils.data import DataLoader, random_split
from torch import optim
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparams = {
    "num_epochs": 15,
    "batch_size": 128,
    "learning_rate": 1e-3,
}


def train(model, train_loader):
    print("starting train")

    loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    optimizer = optim.Adam(model.parameters(), hyperparams['learning_rate'])

    model = model.train()

    for epoch in range(hyperparams['num_epochs']):
        losses = []
        print(epoch)
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
        print("epoch loss:", np.mean(losses))


def validate(model, validate_loader):
    print("starting validation")
    loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

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

    torch.save(model.state_dict(), './model.pt')
    print("mean loss:", np.mean(losses))


def test(model, test_loader):
    print("starting test")
    model = model.eval()
    classification = []

    for batch in tqdm(test_loader):
        x = batch['x']
        cell_type = batch['cell_type']
        id = batch['id']
        x = x.unsqueeze(1)
        x = x.to(device)

        y_pred = model(x)
        for i in range(y_pred.size()[0]):
            # print(cell_type[i].item(), id[i].item(), y_pred[i].item())
            classification.append((cell_type[i].item() + "_" + str(int(id[i].item())), str(y_pred[i].item())))

    df = pd.DataFrame(classification, columns=['id', 'expression'])
    df.to_csv('submission.csv', index=False)

# nohup python histone.py -s -T data/train.npz -t data/eval.npz &
# python histone.py -s -T data/train.npz -t data/eval.npz
# python histone.py -l -T data/train.npz -t data/eval.npz
# python histone.py -l ./data -t data/eval.npz
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
    seq_file = 'data/seq_data.csv'

    model = densenet().to(device)
    print("gathering train data")
    train_loader = None
    if args.train:
        train_file = args.train[0]
        dataset = HistoneDataset(train_file, seq_file)

        split_amount = int(len(dataset) * 0.9)

        train_dataset, validate_dataset = random_split(
            dataset, (split_amount, len(dataset) - split_amount))

        train_loader = DataLoader(
            train_dataset, batch_size=hyperparams['batch_size'], shuffle=True
        )
        validate_loader = DataLoader(
            validate_dataset, batch_size=hyperparams['batch_size'], shuffle=True
        )

    print("gathering test data")
    test_loader = None
    if args.test:
        test_file = args.test[0]
        test_dataset = HistoneDataset(test_file, seq_file)
        test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt', map_location=torch.device(device)))
    if args.train:
        print("running training loop...")
        train(model, train_loader)
        validate(model, validate_loader)
    if args.test:
        print("running testing loop...")
        test(model, test_loader)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
