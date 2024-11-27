import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import scipy


class GeneDataset(Dataset):
    def __init__(self, adata, target=None):
        genes = adata.X
        if scipy.sparse.issparse(genes):
            genes = genes.toarray()
        self.genes = torch.tensor(genes, dtype=torch.float32)
        self._label_encoder = None
        self.target = target
        self.labels = torch.zeros(genes.shape[0], dtype=torch.int64)
        if self.target:
            labels = np.array(adata.obs[target])
            if labels.dtype != "int":
                le = LabelEncoder()
                labels = le.fit_transform(labels)
                self._label_encoder = le
            self.labels = torch.tensor(labels, dtype=torch.int64)

        self.n_samples = genes.shape[0]
        self.n_genes = genes.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.genes[idx, :], self.labels[idx]
