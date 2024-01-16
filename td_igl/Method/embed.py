import torch


def embed(input, basis):
    # print(input.shape, basis.shape)
    projections = torch.einsum("bnd,de->bne", input, basis)  # .permute(2, 0, 1)
    # print(projections.max(), projections.min())
    embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
    return embeddings  # B x N x E
