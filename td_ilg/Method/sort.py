import torch


def sortCenters(centers_quantized, encodings):
    ind3 = torch.argsort(centers_quantized[:, :, 2], dim=1)
    centers_quantized = torch.gather(
        centers_quantized,
        1,
        ind3[:, :, None].expand(-1, -1, centers_quantized.shape[-1]),
    )
    encodings = torch.gather(encodings, 1, ind3)

    _, ind2 = torch.sort(centers_quantized[:, :, 1], dim=1, stable=True)
    centers_quantized = torch.gather(
        centers_quantized,
        1,
        ind2[:, :, None].expand(-1, -1, centers_quantized.shape[-1]),
    )
    encodings = torch.gather(encodings, 1, ind2)

    _, ind1 = torch.sort(centers_quantized[:, :, 0], dim=1, stable=True)
    centers_quantized = torch.gather(
        centers_quantized,
        1,
        ind1[:, :, None].expand(-1, -1, centers_quantized.shape[-1]),
    )
    encodings = torch.gather(encodings, 1, ind1)
    return centers_quantized, encodings
