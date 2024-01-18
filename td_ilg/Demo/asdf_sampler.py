import sys

sys.path.append("../data-convert/")
sys.path.append("../spherical-harmonics/")
sys.path.append("../a-sdf/")


def demo():
    from td_ilg.Module.asdf_sampler import ASDFSampler

    asdf_sampler = ASDFSampler()
    asdf_sampler.sample()
    return True
