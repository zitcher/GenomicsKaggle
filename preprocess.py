import argparse
import pprint

import numpy as np
import torch
from torch.utils.data import Dataset


class HistoneDataset(Dataset):
    def __init__(self, input_file):
        """
        :param input_file: the data file pathname
        """
        self.pp = pprint.PrettyPrinter(indent=4)
        # TODO: read the input file line by line and put the lines in a list.
        npzfile = np.load(input_file)

        # [50, 16000, 100, 7]
        # [cell_types, genes, bins, (columns)]
        # columns = GeneID, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3, Expression Value (same for entire bin)
        # columns 0: GeneId, 1-5: Histone Marks, 6: Expression Value
        self.npdata = npzfile
        self.cell_types = npzfile.files

        # pp.pprint(npzfile[self.cell_types[0]][0])

        self.x = []
        self.y = []

        # [cell_types, genes, bins, histomes]
        input = []
        # [cell_types, genes, expression]
        output = []

        for cell in self.cell_types:
            cell_data = self.npdata[cell]
            hm_data = cell_data[:, :, 1:6]
            exp_values = cell_data[:, 0, 6]
            input.append(hm_data)
            output.append(exp_values)

        # [cell_types*genes, bins, histomes]
        input = np.concatenate(input, axis=0)
        # [cell_types*genes, expression]
        output = np.concatenate(output, axis=0)

        for x in input:
            self.x.append(torch.tensor(x))

        for y in output:
            self.y.append(torch.tensor(y))

        # width corresponds to columns
        self.width = self.x[0].size()[1].item()

        # height corresponds to rows
        self.heigth = self.x[0].size()[0].item()

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # TODO: Override method to return length of dataset
        return len(self.y)

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index
        In this case, you should return your original and target sentence and
        anything else you may need in training/validation/testing.

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the items in dataset
        item = {
            "x": self.x[idx],
            "y": self.y[idx],
        }
        return item


if __name__ == '__main__':

    pp = pprint.PrettyPrinter(indent=4)
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input dataset')
    args = parser.parse_args()

    dataset = HistoneDataset(args.input_file)
    pp.pprint(dataset)
