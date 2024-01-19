import sys

sys.path.append("../data-convert/")
sys.path.append("../spherical-harmonics/")
sys.path.append("../a-sdf/")


def demo():
    from td_ilg.Module.asdf_autoencoder_sampler import ASDFAutoEncoderSampler

    model_file_path = "/Users/fufu/Nutstore Files/paper-materials-ASDF/Model/lr_1e-2_1batch/model_best.pth"
    device = "cpu"
    mini_dataset_folder_path = (
        "/Users/fufu/Nutstore Files/paper-materials-ASDF/Dataset/mini/"
    )
    mesh_file_path_list = [
        mini_dataset_folder_path + "03001627/26bee1a8ea71545c3a288f3e01ebe3.obj",
        mini_dataset_folder_path + "03001627/40ee8ed17f6ea51224669056e0d19a1.obj",
    ]
    sample_point_num = 4000
    rad_density = 10

    asdf_autoencoder_sampler = ASDFAutoEncoderSampler(model_file_path, device)
    asdf_autoencoder_sampler.sample(
        mesh_file_path_list, sample_point_num, rad_density)
    return True
