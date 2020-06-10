import torch.nn as nn
import torch
from blocks import *


class VQVAE(nn.Module):
    def __init__(self, num_hiddens=128, num_residual_hiddens=32,
                 num_residual_layers=2, embedding_dim=64, num_embeddings=512,
                 in_channels=1, beta=0.25, use_gpu=True):
        super(VQVAE, self).__init__()
        self.beta = beta

        self.embedding_dim = embedding_dim

        self.encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.last_conv = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

        # self.codebook=torch.randn(num_embeddings, embedding_dim)#initialize with gaussian KxD matrix
        self.codebook = torch.zeros(num_embeddings, embedding_dim)  # initialize with zeros KxD matrix
        self.codebook.requires_grad_()
        if use_gpu:
            self.codebook = self.codebook.cuda()
        self.codebook.retain_grad()
        self.codebook = nn.Parameter(self.codebook)

    def forward(self, x):
        ze_hat = self.encoder(x)
        self.ze = self.last_conv(ze_hat)
        self.ze.retain_grad()

        ze_tmp = self.ze.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)  # NxD
        distances = torch.cdist(ze_tmp, self.codebook)  # NxK
        self.indices = torch.argmin(distances, dim=1)

        self.zq = self.codebook[self.indices.detach()].detach()

        self.zq = self.zq.reshape(self.ze.shape[0], self.ze.shape[2], self.ze.shape[3], -1)  # .shape
        self.zq = self.zq.permute(0, 3, 1, 2)
        self.zq.requires_grad_()
        self.zq.retain_grad()

        zq_tmp = self.codebook[self.indices]
        zq_tmp = zq_tmp.reshape(self.ze.shape[0], self.ze.shape[2], self.ze.shape[3], -1)
        zq_tmp = zq_tmp.permute(0, 3, 1, 2)

        L_emb = ((self.zq.detach() - self.ze) ** 2).mean()
        l_commit = self.beta * ((zq_tmp - self.ze.detach()) ** 2).mean()

        out_dec = self.decoder(self.zq)

        return out_dec, L_emb, l_commit

    def compute_out_from_indices(self, indices):
        # indices: B,H_z,W_ze
        shape = indices.shape
        indices = indices.reshape(-1, )
        zq = self.codebook[indices]

        zq = zq.reshape(shape[0], shape[1], shape[2], -1)
        zq = zq.permute(0, 3, 1, 2)
        out = self.decoder(zq)

        return out

    def compute_gradients(self, x):
        out, l_emb, l_commit = self.forward(x)

        #####updates encoder and decoder params###
        L_likelihood = ((x - out) ** 2).mean()
        L_likelihood.backward()

        # copy gradients from zq to ze
        self.ze.backward(self.zq.grad, retain_graph=True)
        #################################

        ####updates only encoder using embeddings####
        l_emb.backward()
        ##########################

        #####updates only the embeddings####
        l_commit.backward()
        ###################################
        return L_likelihood + l_emb + l_commit, out