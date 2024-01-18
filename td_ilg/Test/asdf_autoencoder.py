import sys

sys.path.append("../data-convert/")
sys.path.append("../spherical-harmonics/")
sys.path.append("../a-sdf/")


def test():
    import torch
    from td_ilg.Model.asdf_autoencoder import ASDFAutoEncoder

    asdf_channel = 40
    sh_2d_degree = 3
    sh_3d_degree = 6
    hidden_dim = 128
    batch_size = 1

    points = torch.rand(batch_size, 2344, 3)
    idxs = torch.randint(0, 2344, [batch_size, asdf_channel])

    asdf_autoencoder = ASDFAutoEncoder(
        asdf_channel, sh_2d_degree, sh_3d_degree, hidden_dim
    )

    asdf_points = asdf_autoencoder(points)
    asdf_points = asdf_autoencoder(points, idxs)
    print(asdf_points.shape)
    return True
