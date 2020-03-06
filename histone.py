from preprocess import HistoneDataset
from model import Linear
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
from sklearn.tree import DecisionTreeRegressor
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, HistGradientBoostingRegressor
import numpy as np
from joblib import dump, load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, hyperparams):
    print("starting train")

    X = []
    y = []

    for batch in tqdm(train_loader):
        X.append(batch['x'].flatten().numpy())
        y.append(batch['y'].numpy()[0])

    model.fit(X, y)



# def validate(model, validate_loader, hyperparams):
#     print("starting validation")
#     mse = []
#
#     for batch in tqdm(validate_loader):
#         y = batch['y'].numpy()[0]
#         predict = model.predict([batch['x'].flatten().numpy()])[0]
#         # print("predict", predict, "y", y)
#         score = (predict - y) ** 2
#         mse.append(score)
#
#     print("Validation MSE", np.mean(mse))
def validate(model, validate_loader, hyperparams):
    print("starting validation")
    X = []
    y = []
    mse = []
    for batch in tqdm(validate_loader):
        X.append(batch['x'].flatten().numpy())
        y.append(batch['y'].numpy()[0])

    predictions = model.predict(X)

    for i in range(len(predictions)):
        score = (predictions[i] - y[i]) ** 2
        mse.append(score)

    print("Validation MSE", np.mean(mse))

def test(model, test_loader, hyperparams):
    print("starting test")
    classification = []
    for batch in tqdm(test_loader):
        cell_type = batch['cell_type']
        id = batch['id']
        predict = model.predict([batch['x'].flatten().numpy()])[0]
        classification.append((cell_type[0].item() + "_" + str(int(id[0].item())), str(predict)))

    df = pd.DataFrame(classification, columns=['id', 'expression'])
    df.to_csv('submission1.csv', index=False)


# python histone.py -s -S ./data -T data/train.npz -t data/eval.npz
# python histone.py -s -L ./data -T data/train.npz -t data/eval.npz
# python histone.py -lL ./data -t data/eval.npz
# /home/zachary_j_hoffman/DLGenomics/submission.csv
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
        "batch_size": 1,
    }

    # rng = np.random.RandomState(1)
    # model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=2000, random_state=rng)
    # model = RandomForestRegressor(n_estimators=500, max_depth=2, random_state=0)
    model = HistGradientBoostingRegressor()
    train_dataset = None
    validate_dataset = None
    test_dataset = None


    if args.train:
        train_file = args.train[0]
        dataset = HistoneDataset(train_file, args.savedata, args.loaddata, "train")

        split_amount = int(len(dataset) * 0.8)

        train_dataset, validate_dataset = random_split(
            dataset, (split_amount, len(dataset) - split_amount))

    if args.test:
        test_file = args.test[0]

        test_dataset = HistoneDataset(test_file, args.savedata, args.loaddata, "eval")

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
        model = load('./model.pt')
    if args.train:
        print("running training loop...")
        train(model, train_loader, hyperparams)
        validate(model, validate_loader, hyperparams)
    if args.test:
        print("running testing loop...")
        test(model, test_loader, hyperparams)
    if args.save:
        print("saving model...")
        dump(model, './model.pt')
