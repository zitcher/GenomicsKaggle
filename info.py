import numpy as np

# Keys to npzfile of train & eval
train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058', 
'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097', 
'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

# Load data
train_data = np.load('train.npz')
eval_data = np.load('eval.npz')

# Combine Train Data to use information from all cells
train_inputs = [] # Input histone mark data
train_outputs = [] # Correct expression value
for cell in train_cells:
    cell_data = train_data[cell]
    hm_data = cell_data[:,:,1:6]
    exp_values = cell_data[:,0,6]
    train_inputs.append(hm_data)
    train_outputs.append(exp_values)

train_inputs = np.concatenate(train_inputs, axis=0)
train_outputs = np.concatenate(train_outputs, axis=0)

# Prepare Eval inputs in similar way
eval_inputs = []
for cell in eval_cells:
    cell_data = eval_data[cell]
    hm_data = cell_data[:,:,1:6]
    eval_inputs.append(hm_data)

eval_inputs = np.concatenate(eval_inputs, axis=0)


# Example submission of predicted values
import pandas as pd

cell_list = []
gene_list = []
example_eval_preds = np.random.randn(len(eval_inputs))
for cell in eval_cells:
    cell_data = eval_data[cell]
    cell_list.extend([cell]*len(cell_data))
    genes = cell_data[:,0,0].astype('int32')
    gene_list.extend(genes)

id_column = [] # ID is {cell type}_{gene id}
for idx in range(len(eval_inputs)):
    id_column.append(f'{cell_list[idx]}_{gene_list[idx]}')

df_data = {'id': id_column, 'expression' : example_eval_preds}
submit_df = pd.DataFrame(data=df_data)

submit_df.to_csv('sample_submission.csv', header=True, index=False, index_label=False)