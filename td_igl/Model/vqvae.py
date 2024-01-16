import torch

from td_igl.Model.VQVAE.auto_encoder import AutoEncoder


def vqvae_64_1024_2048(pretrained=False, **kwargs):
    model = AutoEncoder(N=64, K=1024, M=2048, **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def vqvae_128_1024_2048(pretrained=False, **kwargs):
    model = AutoEncoder(N=128, K=1024, M=2048, **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def vqvae_256_1024_2048(pretrained=False, **kwargs):
    model = AutoEncoder(N=256, K=1024, M=2048, **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def vqvae_512_1024_2048(pretrained=False, **kwargs):
    model = AutoEncoder(N=512, K=1024, M=2048, **kwargs)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
