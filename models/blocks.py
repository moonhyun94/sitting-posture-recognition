import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):

    def __init__(self, input_dim, latent_dim, patch_size):
        super(PatchEmbedding, self).__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(input_dim, latent_dim, (patch_size, patch_size), stride=patch_size, padding=0),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        """
        Arguments:
        ----------
        - x: images (N, C, H, W)
        """

        return self.embedding(x)


class PatchDecoding(nn.Module):

    def __init__(self, output_dim, latent_dim, patch_size):
        super(PatchDecoding, self).__init__()

        self.embedding = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, output_dim, (patch_size, patch_size), stride=patch_size, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Arguments:
        ----------
        - x: images (N, C, H, W)
        """

        return self.embedding(x)


class DepthwiseMixer(nn.Module):

    def __init__(self, latent_dim, kernel_size=7):
        super(DepthwiseMixer, self).__init__()

        self.depthwise_mixer = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, (kernel_size, kernel_size), stride=1, padding=kernel_size//2, groups=latent_dim),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        """
        Arguments:
        ----------
        - x: features (N, C, H, W)
        """

        return self.depthwise_mixer(x)


class DepthwiseTransposeMixer(nn.Module):

    def __init__(self, latent_dim, kernel_size=7):
        super(DepthwiseTransposeMixer, self).__init__()

        self.depthwise_mixer = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, (kernel_size, kernel_size), stride=1, padding=kernel_size//2, groups=latent_dim),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        """
        Arguments:
        ----------
        - x: features (N, C, H, W)
        """

        return self.depthwise_mixer(x)


class PointwiseMixer(nn.Module):

    def __init__(self, latent_dim):
        super(PointwiseMixer, self).__init__()

        self.pointwise_mixer = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        """
        Arguments:
        ----------
        - x: features (N, C, H, W)
        """
        
        return self.pointwise_mixer(x)


class PointwiseTransposeMixer(nn.Module):

    def __init__(self, latent_dim):
        super(PointwiseTransposeMixer, self).__init__()

        self.pointwise_mixer = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        """
        Arguments:
        ----------
        - x: features (N, C, H, W)
        """
        
        return self.pointwise_mixer(x)


class ConvMixerBlock(nn.Module):

    def __init__(self, latent_dim, kernel_size=7):
        super(ConvMixerBlock, self).__init__()

        self.depthwise_mixer = DepthwiseMixer(latent_dim, kernel_size)
        self.pointwise_mixer = PointwiseMixer(latent_dim)

        self.embedder = nn.Sequential(
            nn.Conv2d(latent_dim*2, latent_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        f = self.pointwise_mixer(x)
        f = self.depthwise_mixer(f)
        f = torch.cat([f, x], dim=1)
        f = self.embedder(f)
        return f


class ConvTransposeMixerBlock(nn.Module):

    def __init__(self, latent_dim, kernel_size=7):
        super(ConvTransposeMixerBlock, self).__init__()

        self.depthwise_mixer = DepthwiseTransposeMixer(latent_dim, kernel_size)
        self.pointwise_mixer = PointwiseTransposeMixer(latent_dim)

    def forward(self, x):
        f = self.pointwise_mixer(x)
        f = self.depthwise_mixer(f) + x
        return f
