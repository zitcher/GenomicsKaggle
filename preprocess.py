from torch.utils.data import Dataset
import torch
import numpy as np
import argparse
import pprint


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
        # columns = GeneID, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3, Expression Value
        # columns 0: GeneId, 1-5: Histone Marks, 6: Expression Value
        self.npdata = npzfile
        self.cell_types = npzfile.files

        pp.pprint(npzfile[self.cell_types[0]].shape)

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
