import argparse
import pprint

import numpy as np
import torch
from torch.utils.data import Dataset


class HistoneDataset(Dataset):
    def __init__(self, input_file, save_path=None, load_path=None, mode="train"):
        """
        :param input_file: the data file pathname
        """
        self.pp = pprint.PrettyPrinter(indent=4)
        npzfile = np.load(input_file)

        # [50, 16000, 100, 7]
        # [cell_types, genes, bins, (columns)]
        # columns = GeneID, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3, Expression Value (same for entire bin)
        # columns 0: GeneId, 1-5: Histone Marks, 6: Expression Value
        self.npdata = npzfile
        self.cell_types = npzfile.files

        # pp.pprint(npzfile[self.cell_types[0]][0])

        train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058',
                       'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

        eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097',
                      'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']
        all_cells = dict()
        for cell in train_cells:
            if cell not in all_cells:
                all_cells[cell] = len(all_cells)

        for cell in eval_cells:
            if cell not in all_cells:
                all_cells[cell] = len(all_cells)

        # self.pp.pprint(all_cells)

        # [cell_types, genes, bins, histomes]
        input = []
        # [cell_types, genes, expression]
        output = []
        # [cell_types, genes, expression]
        ids = []
        # type
        type = []
        cell_nums = []

        if load_path is not None:
            input = np.load(load_path[0] + "/" + mode + "/x.npy")
            output = np.load(load_path[0] + "/" + mode + "/y.npy")
            ids = np.load(load_path[0] + "/" + mode + "/id.npy")
            type = np.load(load_path[0] + "/" + mode + "/type.npy")
        else:
            for cell in self.cell_types:
                cell_data = self.npdata[cell]
                cell_num = all_cells[cell]
                id = cell_data[:, 0, 0]
                hm_data = cell_data[:, :, 1:6]
                exp_values = cell_data[:, 0, 6]  # cell_data[:, 0, 6]
                ids.append(id)
                input.append(hm_data)
                output.append(exp_values)
                cell_nums.extend([cell_num] * cell_data.shape[0])
                type.extend([cell] * cell_data.shape[0])

            # [cell_types*genes, seq_len (bin), embed (histones)]
            input = np.concatenate(input, axis=0)
            # [cell_types*genes]
            output = np.concatenate(output, axis=0)
            # [cell_types*genes]
            ids = np.concatenate(ids, axis=0)
            # [cell_types*genes]
            type = np.asarray(type)

            if save_path is not None:
                np.save(save_path[0] + "/" + mode + "/x", input)
                np.save(save_path[0] + "/" + mode + "/y", output)
                np.save(save_path[0] + "/" + mode + "/id", ids)
                np.save(save_path[0] + "/" + mode + "/type", type)

        self.x = []
        self.y = []
        self.cell_nums = []
        self.id = ids
        self.type = type

        # [cell_types*genes, seq_len (bin), embed (histones)]
        for x in input:
            self.x.append(torch.tensor(x))

        # [cell_types*genes, seq_len (bin)]
        for y in output:
            self.y.append(torch.tensor(y))

        # [cell_types*genes]
        for cell in cell_nums:
            self.cell_nums.append(torch.tensor(cell))

        # width corresponds to columns
        self.width = self.x[0].size()[1]

        # height corresponds to rows - 100
        self.height = self.x[0].size()[0]

        # print('width', self.width)
        # print('height', self.height)
        #
        # self.pp.pprint(self.x[1])
        # self.pp.pprint(self.y[1])



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
            "cell_type": self.type[idx],
            "cell_num": self.cell_nums[idx],
            "id": self.id[idx],
            "x": self.x[idx],
            "y": self.y[idx],
            # "width": self.width,
            # "height": self.height
        }
        return item


if __name__ == '__main__':

    pp = pprint.PrettyPrinter(indent=4)
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input dataset')
    args = parser.parse_args()

    dataset = HistoneDataset(args.input_file)
    pp.pprint(dataset)
