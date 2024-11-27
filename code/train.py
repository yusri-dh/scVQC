from data_loader import *
from model import *
from visualization import *


def train_vqvae(
    train_set,
    batch_size=256,
    device=torch.device("cuda"),
    learning_rate=1e-3,
    n_epochs=10,
    lr_weight_decay=1e-3,
    hidden_dim1=512,
    hidden_dim2=128,
    num_embeddings=100,
    embedding_dim=32,
    commitment_cost=0.5,
    verbose=True,
    split=10,
    fsq=False,
    levels=None,
    ema=True,
    decay=0.99,
    epsilon=1e-5,
    dropout=0.1,
):
    train_loader = DataLoader(
        train_set, batch_size=batch_size, drop_last=True, shuffle=True
    )
    vqvae = VQVAE(
        input_dim=train_set.n_genes,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
        split=split,
        fsq=fsq,
        levels=levels,
        ema=ema,
        decay=decay,
        epsilon=epsilon,
        dropout=dropout,
    ).to(device)
    vqvae.train()
    train_res_recon_error = []
    train_res_vq_loss = []
    train_res_total_loss = []
    train_res_perplexity = []
    optimizer = optim.Adam(
        vqvae.parameters(),
        lr=learning_rate,
        weight_decay=lr_weight_decay,
        amsgrad=False,
    )
    i = 0
    for epoch in range(n_epochs):
        for X_batch, _ in train_loader:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            batch_size = X_batch.shape[0]
            vq_loss, X_recon, perplexity, _ = vqvae(X_batch.view(batch_size, -1))
            recon_error = F.mse_loss(X_recon, X_batch)
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_vq_loss.append(vq_loss.item())
            train_res_total_loss.append(loss.item())
            train_res_perplexity.append(perplexity.item())
            i += 1
            if verbose:
                if i % 500 == 0:
                    print("%d iterations" % i)
                    print("recon_error: %.5f" % np.mean(train_res_recon_error[-500:]))
                    print("vq_loss: %.5f" % np.mean(train_res_vq_loss[-500:]))

                    print("perplexity: %.5f" % np.mean(train_res_perplexity[-500:]))
    print("Finished VQ-VAE training")
    return (
        vqvae,
        train_res_recon_error,
        train_res_vq_loss,
        train_res_total_loss,
        train_res_perplexity,
    )


def train_vqclassifier(
    train_set,
    n_class,
    batch_size=256,
    device=torch.device("cuda"),
    learning_rate=1e-3,
    n_epochs=10,
    lr_weight_decay=0,
    hidden_dim1=1024,
    hidden_dim2=256,
    num_embeddings=200,
    embedding_dim=100,
    commitment_cost=0.25,
    verbose=True,
    split=20,
    fsq=False,
    levels=None,
    ema=True,
    decay=0.99,
    epsilon=1e-5,
    dropout=0.1,
):
    train_loader = DataLoader(
        train_set, batch_size=batch_size, drop_last=True, shuffle=True
    )
    vqclassifier = VQClassifier(
        input_dim=train_set.n_genes,
        n_class=n_class,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
        split=split,
        fsq=fsq,
        levels=levels,
        ema=ema,
        decay=decay,
        epsilon=epsilon,
        dropout=dropout,
    ).to(device)
    vqclassifier.train()
    train_res_prediction_error = []
    train_res_vq_loss = []
    train_res_total_loss = []
    train_res_perplexity = []
    optimizer = optim.Adam(
        vqclassifier.parameters(),
        lr=learning_rate,
        weight_decay=lr_weight_decay,
        amsgrad=False,
    )
    i = 0
    for epoch in range(n_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = X_batch.shape[0]
            vq_loss, prediction, perplexity, _ = vqclassifier(
                X_batch.view(batch_size, -1)
            )
            # recon_error = F.mse_loss(X_recon, X_batch)
            pred_loss = F.nll_loss(prediction, y_batch)
            loss = pred_loss + vq_loss
            loss.backward()

            optimizer.step()

            train_res_prediction_error.append(pred_loss.item())
            train_res_vq_loss.append(vq_loss.item())
            train_res_total_loss.append(loss.item())
            train_res_perplexity.append(perplexity.item())
            i += 1
            if verbose:
                if i % 500 == 0:
                    print("%d iterations" % i)
                    print(
                        "prediction_loss: %.5f"
                        % np.mean(train_res_prediction_error[-500:])
                    )
                    print("vq_loss: %.5f" % np.mean(train_res_vq_loss[-500:]))

                    print("perplexity: %.5f" % np.mean(train_res_perplexity[-500:]))
    print("Finished VQ-VAE training")
    return (
        vqclassifier,
        train_res_prediction_error,
        train_res_vq_loss,
        train_res_total_loss,
        train_res_perplexity,
    )


def train(
    train_set,
    supervised=True,
    n_class=None,
    batch_size=32,
    device=torch.device("cuda"),
    learning_rate=5e-4,
    n_epochs=5,
    lr_weight_decay=1e-6,
    hidden_dim1=1024,
    hidden_dim2=256,
    num_embeddings=200,
    embedding_dim=100,
    commitment_cost=0.25,
    verbose=True,
    split=20,
    fsq=False,
    levels=None,
    ema=True,
    decay=0.99,
    epsilon=1e-5,
    dropout=0.1,
):
    if supervised:
        (
            model,
            train_res_prediction_error,
            train_res_vq_loss,
            train_res_total_loss,
            train_res_perplexity,
        ) = train_vqclassifier(
            train_set,
            n_class,
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
            split=split,
            fsq=fsq,
            levels=levels,
            ema=ema,
            decay=decay,
            epsilon=epsilon,
            dropout=dropout,
        )
    else:
        (
            model,
            train_res_prediction_error,
            train_res_vq_loss,
            train_res_total_loss,
            train_res_perplexity,
        ) = train_vqvae(
            train_set,
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
            split=split,
            fsq=fsq,
            levels=levels,
            ema=ema,
            decay=decay,
            epsilon=epsilon,
            dropout=dropout,
        )
    return (
        model,
        train_res_prediction_error,
        train_res_vq_loss,
        train_res_total_loss,
        train_res_perplexity,
    )
