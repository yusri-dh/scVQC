import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import umap
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, FunctionTransformer

# from adjustText import adjust_text
import seaborn as sns

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)


def visualize_loss(
    train_res_prediction_error,
    train_res_vq_loss,
    train_res_total_loss,
    train_res_perplexity,
    save_path="./out_loss.png",
):
    train_res_recon_error_smooth = savgol_filter(train_res_prediction_error, 200, 7)
    train_res_vq_loss_smooth = savgol_filter(train_res_vq_loss, 200, 7)
    train_res_total_loss_smooth = savgol_filter(train_res_total_loss, 200, 7)
    train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 200, 7)

    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(2, 3, 1)
    ax.plot(train_res_recon_error_smooth)
    # ax.set_yscale('log')
    ax.set_title("Smoothed NMSE.")
    ax.set_xlabel("iteration")

    ax = f.add_subplot(2, 3, 2)
    ax.plot(train_res_vq_loss_smooth)
    # ax.set_yscale('log')
    ax.set_title("Smoothed VQ loss.")
    ax.set_xlabel("iteration")

    ax = f.add_subplot(2, 3, 3)
    ax.plot(train_res_total_loss_smooth)
    # ax.set_yscale('log')
    ax.set_title("Smoothed total loss.")
    ax.set_xlabel("iteration")

    ax = f.add_subplot(2, 3, 4)
    ax.plot(train_res_perplexity_smooth)
    ax.set_title("Smoothed Average codebook usage (perplexity).")
    ax.set_xlabel("iteration")

    f.tight_layout()

    f.savefig(save_path, bbox_inches="tight")
    return f


def visualize_embedding(
    model,
    X_train,
    y_train,
    device="cuda",
    sampling=0,
    s=10,
    save_path="./out_embedding.png",
):

    y_train = np.array(y_train)
    le = None
    if y_train.dtype != "int":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
    if sampling > 0:
        indices = np.random.choice(X_train.shape[0], sampling, replace=False)
        X_train = X_train[indices, :]
        y_train = y_train[indices]
    with torch.no_grad():
        model.eval()
        model_embeddings, model_quantized, _ = model.embed(
            torch.tensor(X_train, dtype=torch.float32).to(device)
        )
    model_embeddings = model_embeddings.cpu().detach().numpy()
    model_quantized = model_quantized.cpu().detach().numpy()

    # use no random seed for umap for parallelization
    umap_model_embeddings = umap.UMAP().fit_transform(model_embeddings)
    umap_model_quantized = umap.UMAP().fit_transform(model_quantized)
    # umap_model_encodings = umap.UMAP(metric="jaccard").fit_transform(model_encodings)

    f = plt.figure(figsize=(25, 10))
    ax1 = f.add_subplot(1, 2, 1)
    ax2 = f.add_subplot(1, 2, 2)
    # ax3 = f.add_subplot(1, 3, 3)

    ax2_texts = []
    unique_target = np.unique(y_train)
    if len(unique_target) <= 20:
        cmaps = sns.color_palette("tab20")
    else:
        cmaps = sns.color_palette("husl", len(unique_target))
    for y_ in unique_target:
        idx = y_train == y_
        # ax.scatter(umap_embedding[idx, 0], umap_embedding[idx, 1], s=0.1,label=cell_type_dict[y_])
        if le:
            label = le.inverse_transform([y_])

        else:
            label = y_

        ax1.scatter(
            umap_model_embeddings[idx, 0],
            umap_model_embeddings[idx, 1],
            color=cmaps[y_],
            label=label,
            s=s,
        )
        ax2.scatter(
            umap_model_quantized[idx, 0],
            umap_model_quantized[idx, 1],
            color=cmaps[y_],
            label=label,
            s=s,
        )
        # ax3.scatter(
        #     umap_model_encodings[idx, 0],
        #     umap_model_encodings[idx, 1],
        #     color=cmaps[y_],
        #     label=label,
        #     s=3,
        # )

        # ax2_texts.append(
        #     ax2.annotate(
        #         label,
        #         (
        #             umap_model_quantized[y_, 0],
        #             umap_model_quantized[y_, 1],
        #         ),
        #         arrowprops=dict(arrowstyle="->"),
        #     )
        # )
    # adjust_text(ax2_texts)
    ax2.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    ax1.set_title("Embedding")
    ax2.set_title("Quantized")
    # ax3.set_title("Encoding jaccard")

    f.tight_layout()
    f.savefig(save_path, bbox_inches="tight")
    return f
