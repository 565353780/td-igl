import torch

from td_ilg.Model.asdf_encoder import ASDFEncoder


def test():
    asdf_channel = 40
    sh_2d_degree = 3
    sh_3d_degree = 6
    hidden_dim = 128
    batch_size = 2

    test_data = torch.rand(batch_size, 2344, 3)
    test_idxs = torch.randint(0, 2344, [batch_size, asdf_channel])

    asdf_encoder = ASDFEncoder(asdf_channel, sh_2d_degree, sh_3d_degree, hidden_dim)

    test_output = asdf_encoder(test_data)
    test_output = asdf_encoder(test_data, test_idxs)
    print(test_output.shape)
    return True
