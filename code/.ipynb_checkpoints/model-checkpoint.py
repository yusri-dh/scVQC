import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fsq import *
import warnings

import numpy


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs
        # flat_input = inputs.view(-1,self.embedding_dim)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized, perplexity, encoding_indices


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay=0.99,
        epsilon=1e-5,
    ):
        super(VectorQuantizerEMA, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (
                1 - self.decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)

            self.embedding.weight = nn.Parameter(
                self.ema_w / self.ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encoding_indices


class Split_Quantizer(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        split=10,
        fsq=False,
        ema=True,
        levels=None,
        decay=0.99,
        epsilon=1e-5,
    ):
        super(Split_Quantizer, self).__init__()
        self.num_embeddings = num_embeddings

        self.commitment_cost = commitment_cost
        self.split = split
        self.levels = levels
        self.fsq = fsq
        self.ema = ema
        self.decay = decay
        self.epsilon = epsilon
        self.embedding_split_dim = embedding_dim // split
        self.embedding_dim = self.embedding_split_dim * split

        if self.fsq:
            if self.levels is None:
                self.levels = [5] * self.embedding_split_dim
            else:
                assert len(self.levels) == self.embedding_split_dim
            self.vq = FSQ(self.levels)

        elif ema:
            self.vq = VectorQuantizerEMA(
                self.num_embeddings,
                self.embedding_split_dim,
                self.commitment_cost,
                self.decay,
                self.epsilon,
            )
        else:
            self.vq = VectorQuantizer(
                self.num_embeddings,
                self.embedding_split_dim,
                self.commitment_cost,
            )

    def forward(self, inputs):
        input_shape = inputs.shape
        loss, quantized, perplexity, encoding_indices = self.vq(
            inputs.reshape(-1, self.embedding_split_dim)
        )
        quantized = quantized.reshape(input_shape[0], -1)
        encoding_indices = encoding_indices.reshape(input_shape[0], -1)
        return loss, quantized, perplexity, encoding_indices


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.linear2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = torch.nn.Linear(hidden_dim2, output_dim)

    def forward(self, X):
        X = self.linear1(X)
        X = F.relu(X)
        X = self.linear2(X)
        X = F.relu(X)
        X = self.linear3(X)

        return X


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.linear2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = torch.nn.Linear(hidden_dim2, output_dim)
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim1)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_dim2)
        self.dropout = dropout

    def forward(self, X):
        X = self.linear1(X)
        X = F.dropout(X, p=self.dropout)
        X = F.relu(X)
        X = self.batchnorm1(X)
        X = self.linear2(X)
        X = F.dropout(X, p=self.dropout)
        X = F.relu(X)
        X = self.batchnorm2(X)
        X = self.linear3(X)

        return X


class Classifier(nn.Module):
    def __init__(self, input_dim, n_class):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, n_class)

    def forward(self, X):
        X = self.linear1(X)
        # X = F.relu(X)
        # X = self.linear2(X)
        X = F.log_softmax(X, dim=-1)
        return X


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim1,
        hidden_dim2,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        split=10,
        fsq=False,
        ema=True,
        levels=None,
        decay=0.99,
        epsilon=1e-5,
        dropout=0.1,
    ):
        super(VQVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_embeddings = num_embeddings

        self.embedding_split_dim = embedding_dim // split
        # to ensure the embedding dimension is a multiplication of the n_split
        self.embedding_dim = self.embedding_split_dim * split
        self.commitment_cost = commitment_cost
        self.split = split
        self.fsq = fsq
        self.levels = levels
        self.ema = ema
        self.decay = decay
        self.epsilon = epsilon
        self.dropout = dropout

        enc_input_dim, dec_output_dim = input_dim, input_dim
        enc_hidden_dim1, dec_hidden_dim2 = hidden_dim1, hidden_dim1
        enc_hidden_dim2, dec_hidden_dim1 = hidden_dim2, hidden_dim2
        enc_output_dim, dec_input_dim = embedding_dim, embedding_dim

        self.encoder = Encoder(
            enc_input_dim,
            enc_hidden_dim1,
            enc_hidden_dim2,
            enc_output_dim,
            self.dropout,
        )
        self.split_quantizer = Split_Quantizer(
            self.num_embeddings,
            self.embedding_dim,
            self.commitment_cost,
            self.split,
            fsq=self.fsq,
            levels=self.levels,
            ema=self.ema,
            decay=0.99,
            epsilon=1e-5,
        )
        self.decoder = Decoder(
            dec_input_dim, dec_hidden_dim1, dec_hidden_dim2, dec_output_dim
        )

    def forward(self, inputs):
        z = self.encoder(inputs)
        loss, quantized, perplexity, _ = self.split_quantizer(z)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity, quantized

    def embed(self, x):
        z = self.encoder(x)
        _, quantized, _, encoding_indices = self.split_quantizer(z)
        return z, quantized, encoding_indices

    def reconstruct(self, x):
        z = self.encoder(x)
        _, quantized, _, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon


class VQClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        n_class,
        hidden_dim1,
        hidden_dim2,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        split=10,
        fsq=False,
        levels=None,
        ema=True,
        decay=0.99,
        epsilon=1e-5,
        dropout=0.1,
    ):
        super(VQClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_embeddings = num_embeddings

        self.embedding_split_dim = embedding_dim // split
        # to ensure the embedding dimension is a multiplication of the n_split
        self.embedding_dim = self.embedding_split_dim * split
        self.n_class = n_class
        self.commitment_cost = commitment_cost
        self.split = split
        self.fsq = fsq
        self.levels = levels
        self.ema = ema
        self.decay = decay
        self.epsilon = epsilon
        self.dropout = dropout

        enc_input_dim, dec_output_dim = input_dim, input_dim
        enc_hidden_dim1, dec_hidden_dim2 = hidden_dim1, hidden_dim1
        enc_hidden_dim2, dec_hidden_dim1 = hidden_dim2, hidden_dim2
        enc_output_dim, dec_input_dim = embedding_dim, embedding_dim

        self.encoder = Encoder(
            enc_input_dim,
            enc_hidden_dim1,
            enc_hidden_dim2,
            enc_output_dim,
            self.dropout,
        )
        self.split_quantizer = Split_Quantizer(
            self.num_embeddings,
            self.embedding_dim,
            self.commitment_cost,
            self.split,
            fsq=self.fsq,
            levels=self.levels,
            ema=self.ema,
            decay=0.99,
            epsilon=1e-5,
        )
        self.classifier = Classifier(dec_input_dim, self.n_class)

    def forward(self, inputs):
        z = self.encoder(inputs)
        loss, quantized, perplexity, _ = self.split_quantizer(z)
        prediction = self.classifier(quantized)

        return loss, prediction, perplexity, quantized

    def embed(self, x):
        z = self.encoder(x)
        _, quantized, _, encoding_indices = self.split_quantizer(z)
        return z, quantized, encoding_indices

    def predict(self, x):
        z = self.encoder(x)

        _, quantized, _, _ = self.split_quantizer(z)
        prediction = self.classifier(quantized)
        return prediction
