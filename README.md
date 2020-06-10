# VQVAE_Pytorch
Implementation in Pytorch of [Neural Discrete Representation Learning, Van den Oord et al](https://arxiv.org/pdf/1711.00937.pdf) (VQVAE).
Similar to regular VAE the code optimizes the ELBO, however, instead of using a continuous latent variable **_z_**, the paper proposes to use a discrete vector which is used to index embeddings from a codebook. Also different, the framework learns the prior instead of using a fixed distribution; this is done with a PixelCNN. The code for the pixelCNN used in this repository is my own implementation which is available [here](https://github.com/rogertrullo/Gated-PixelCNN-Pytorch)

