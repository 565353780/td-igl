import torch
import torch.nn.functional as F

from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def sample(logits, top_k=100, top_p=0.85):
    temperature = 1.0
    logits = logits[:, -1, :] / temperature
    # FIXME: current select the best position directly for now
    return torch.sort(logits, descending=True)[1][0, 0].reshape(1, 1)

    probs = F.softmax(logits, dim=-1)

    topk, indices = torch.topk(probs, k=top_k, dim=-1)
    probs = torch.zeros(*probs.shape).to(probs.device).scatter_(1, indices, topk)

    # top-p
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p

    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    probs[indices_to_remove] = 0

    ix = torch.multinomial(probs, num_samples=1)
    return ix
