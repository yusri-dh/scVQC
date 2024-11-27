import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter


# def feature_spectrum(x, y, model):
#     y = np.array(y)
#     le = None
#     if y.dtype != "int":
#         le = LabelEncoder()
#         y = le.fit_transform(y)
#     document_vectors_raw = torch.zeros(len(np.unique(y)), x.shape[1])
#     document_vectors = torch.zeros(len(np.unique(y)), model.num_embeddings)
#     for group in np.unique(y):
#         idx = y == group
#         v = torch.sum(x[idx, :], dim=0)
#         document_vectors_raw[group, :] = v
#         document_vectors[group, :] = v.reshape(-1, model.num_embeddings).sum(dim=0)
#     tf = document_vectors / document_vectors.sum(dim=1, keepdim=True)
#     N_docs = document_vectors.shape[0]
#     df = torch.sum(document_vectors > 0, dim=0)
#     idf = torch.log((N_docs + 1) / (df + 1)) + 1
#     # idf = torch.log(N_docs / (df+1e-10))
#     tfidf = tf * idf
#     return tfidf, le


# def cell_type_spectrum(x, y, model):
#     y = np.array(y)
#     le = None
#     if y.dtype != "int":
#         le = LabelEncoder()
#         y = le.fit_transform(y)

#     cell_type_rep = torch.zeros(len(np.unique(y)), x.shape[1])
#     for group in np.unique(y):
#         idx = y == group
#         v = torch.mean(x[idx, :], dim=0)
#         cell_type_rep[group, :] = v
#     return cell_type_rep, le


def cal_index(indexs, shapes=400):
    ids = []
    for id1 in indexs:
        id3 = np.zeros(shapes, dtype=int)
        id2 = Counter(
            id1.reshape(
                -1,
            )
        )
        for key in id2.keys():
            id3[key] = id2[key]
        ids.append(id3)
    return ids


def cal_tfidf(ys):
    N, M = ys.shape[0], ys.shape[1]
    Ns = np.sum(ys, axis=1).reshape(-1, 1)
    df = np.array([sum(ys[:, i] > 0) for i in range(M)])
    df = np.log(1 + N / df)
    tfidf = (ys / Ns) * df.reshape(1, -1)
    return tfidf


def cal_feat_spe(indices, labels):
    # indices = adata.obsm['feature_index']
    n_codebook = np.max(indices) + 1
    emb_ind = np.array(cal_index(indices, shapes=n_codebook))

    n_sample = len(labels)
    label_unique = np.unique(labels)
    emb_inds0 = np.zeros((len(label_unique), n_codebook))
    for i in label_unique:
        emb_inds0[i] = np.sum(emb_ind[labels == i], axis=0)
    emb_inds = cal_tfidf(emb_inds0)
    print("Feature spectrum shape: ", emb_inds.shape)

    ids = np.argmax(emb_inds, axis=0)
    reodered_ind2 = np.argsort(ids)
    emb_reinds = emb_inds[:, reodered_ind2]

    id_dict = Counter(ids)
    reodered_inds2 = sorted(id_dict.items(), key=lambda x: x[0], reverse=False)
    i0 = np.arange(len(reodered_inds2))
    i1 = 0
    tfidfs = []
    varss = []
    pvalues = []
    for i in range(len(reodered_inds2)):
        i2 = i1 + reodered_inds2[i][1]
        #         print(i,i1,i2)
        i3 = i0[i0 != i]
        vs = emb_reinds[i, i1:i2]
        reodered_ind2[i1:i2] = reodered_ind2[i1:i2][np.argsort(-vs)]
        i1 = i1 + reodered_inds2[i][1]
    emb_reinds = emb_inds[:, reodered_ind2]
    # adata.uns['feature_spectrum'] = emb_reinds

    return emb_reinds
