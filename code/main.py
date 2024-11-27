import argparse
from model_utils import set_seed
from data_loader import *
from model import *
from train import *
from visualization import *
import os
import time

if __name__ == "__main__":
    set_seed(100)
    parser = argparse.ArgumentParser(
        description="<Vector quantized cell representation>"
    )

    parser.add_argument(
        "--dataset_path",
        action="store",
        type=str,
        required=True,
        help="path to the training dataset.",
    )

    parser.add_argument(
        "--supervised",
        action="store_true",
        help="Using supervised classifier model",
    )

    parser.add_argument(
        "--target",
        default="None",
        type=str,
        required=False,
        help="The  column in andata.obs inputs used as label",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        required=False,
        help="GPU Device(s) used for training",
    )

    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        required=False,
        help="batch_size of the training.",
    )

    parser.add_argument(
        "--hidden-dim1",
        default=512,
        type=int,
        required=False,
        help="Dimension of first hidden layer of encoder.",
    )

    parser.add_argument(
        "--hidden-dim2",
        default=128,
        type=int,
        required=False,
        help="Dimension of second hidden layer of encoder.",
    )

    parser.add_argument(
        "--num_embeddings",
        default=50,
        type=int,
        required=False,
        help="Number of codes in the code book.",
    )

    parser.add_argument(
        "--embedding_dim",
        default=30,
        type=int,
        required=False,
        help="The dimension of the embedding",
    )

    parser.add_argument(
        "--split",
        default=10,
        type=int,
        required=False,
        help="The number of the split for split quantization of the embedding",
    )

    parser.add_argument(
        "--fsq",
        action="store_true",
        help="USING VQ (not FSQ) as quantizer",
    )

    parser.add_argument(
        "-l",
        "--levels",
        default=3,
        type=int,
        required=False,
        help="Levels parameters of FSQ",
    )

    parser.add_argument(
        "--no-ema",
        action="store_false",
        help="Train the codebook using gradient descent. Ignored if not using --no-fsq flag.",
    )

    parser.add_argument(
        "--commitment-cost",
        default=0.25,
        type=float,
        required=False,
        help="Commitment cost",
    )

    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        required=False,
        help="Learning rate for training the model",
    )

    parser.add_argument(
        "--ema-decay",
        default=0.99,
        type=float,
        required=False,
        help="Decay of EMA weights",
    )

    parser.add_argument(
        "--ema-epsilon",
        default=1e-5,
        type=float,
        required=False,
        help="The epsilon parameter for updating EMA",
    )

    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        required=False,
        help="The dropout rate of the encoder",
    )

    parser.add_argument(
        "--lr-weight-decay",
        default=1e-6,
        type=float,
        required=False,
        help="Weight Decay of Learning rate",
    )

    parser.add_argument(
        "--n_epochs",
        default=10,
        type=int,
        required=False,
        help="VQVAE Training epochs",
    )

    parser.add_argument(
        "--workers",
        default=64,
        type=int,
        required=False,
        help="number of worker for data loading",
    )
    parser.add_argument(
        "--vis_sample",
        default=0,
        type=int,
        required=False,
        help="Number of cells for UMAP visualization",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        help="visualize Umap embedding plot and loss plot",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the model.",
    )

    args = parser.parse_args()
    print(args)

    target = None if args.target == "None" else args.target
    train_dataset = sc.read_h5ad(args.dataset_path)
    batch_size = args.batch_size
    hidden_dim1 = args.hidden_dim1
    hidden_dim2 = args.hidden_dim2
    num_embeddings = args.num_embeddings
    embedding_dim = args.embedding_dim
    split = args.split
    commitment_cost = args.commitment_cost
    device = args.device
    learning_rate = args.lr
    n_epochs = args.n_epochs
    lr_weight_decay = args.lr_weight_decay
    verbose = False
    split = args.split
    fsq = args.fsq
    levels = [args.levels] * (embedding_dim // split)
    ema = args.no_ema
    decay = args.ema_decay
    epsilon = args.ema_epsilon
    dropout = args.dropout
    supervised = args.supervised
    n_vis_sample = args.vis_sample
    save_model = args.save_model
    train_set = GeneDataset(train_dataset, target=target)
    if supervised:
        assert (
            train_set.target is not None
        ), "The target variable in the train set should be not empty"
        n_class = len(np.unique(train_set.labels))
    else:
        n_class = None
    start_time = time.time()
    (
        model,
        train_res_prediction_error,
        train_res_vq_loss,
        train_res_total_loss,
        train_res_perplexity,
    ) = train(
        train_set,
        supervised=supervised,
        n_class=n_class,
        batch_size=batch_size,
        device=device,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        lr_weight_decay=lr_weight_decay,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
        verbose=verbose,
        fsq=fsq,
        levels=levels,
        split=split,
        ema=ema,
        decay=decay,
        epsilon=epsilon,
        dropout=dropout,
    )

    print(f" - [Training Running time]: {time.time()-start_time}s")

    fname = os.path.splitext(os.path.basename(args.dataset_path))[0]

    model_info = "{}_fsq:{}_ema:{}_supervised:{}_encoder:({}-{}-{})_num_codes:{}_split:{}_commitment_cost:{}_n_epochs:{}_lr:{}".format(
        fname,
        fsq,
        ema,
        supervised,
        hidden_dim1,
        hidden_dim2,
        embedding_dim,
        num_embeddings,
        split,
        commitment_cost,
        n_epochs,
        learning_rate,
    )
    data_dir = model_info
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if save_model:
        model_save_path = data_dir + "/model_" + model_info + ".pt"
        torch.save(model.state_dict(), model_save_path)
        print("Save mode in: " + model_save_path)

    if args.visualize:
        X_train = train_dataset.X
        y_train = 1 if target == "None" else train_dataset.obs[target]
        if scipy.sparse.issparse(X_train):
            X_train = X_train.toarray()
        visualize_loss(
            train_res_prediction_error,
            train_res_vq_loss,
            train_res_total_loss,
            train_res_perplexity,
            save_path=data_dir + "/loss_" + model_info + ".png",
        )
        print("Save loss plot in " + data_dir + "/loss_" + model_info + ".png")

        visualize_embedding(
            model,
            X_train,
            y_train,
            device,
            sampling=n_vis_sample,
            s=0.5,
            save_path=data_dir + "/embedding_" + model_info + ".png",
        )
        print(
            "Save embedding plot in " + data_dir + "/embedding_" + model_info + ".png"
        )
