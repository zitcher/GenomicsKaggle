# DLGenomicsKaggle
# Objective
Predict gene expression based on histone modification

# Papers to Read
Deepchrome: https://academic.oup.com/bioinformatics/article/32/17/i639/2450757

Deephistone: https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-5489-4

# Potential Model

https://arxiv.org/abs/1808.03867

# Contains List of Famous Models You can Reference for Architecture

https://arxiv.org/pdf/1901.06032.pdf

# Preprocess
`python preprocess.py data/train.npz`
`python preprocess.py data/eval.npz`

# Train and Test
`python histone.py -s -T data/train.npz -t data/eval.npz`
