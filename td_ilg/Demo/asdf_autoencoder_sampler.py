import os
import sys

sys.path.append("../data-convert/")
sys.path.append("../spherical-harmonics/")
sys.path.append("../a-sdf/")


def demo():
    from td_ilg.Module.asdf_autoencoder_sampler import ASDFAutoEncoderSampler

    model_file_path = "/Users/fufu/Nutstore Files/paper-materials-ASDF/Model/lr_1e-2_1batch/model_best.pth"
    model_file_path = "/home/chli/Nutstore Files/paper-materials-ASDF/Model/lr_1e-2_1batch/model_best.pth"
    device = "cpu"
    mini_dataset_folder_path = (
        "/Users/fufu/Nutstore Files/paper-materials-ASDF/Dataset/mini/"
    )
    mini_dataset_folder_path = (
        "/home/chli/Nutstore Files/paper-materials-ASDF/Dataset/mini/"
    )
    mesh_file_path_list = [
        mini_dataset_folder_path + "03001627/26bee1a8ea71545c3a288f3e01ebe3.obj",
        mini_dataset_folder_path + "03001627/40ee8ed17f6ea51224669056e0d19a1.obj",
        mini_dataset_folder_path + "02691156/7526757d0fdf8acc14f1e6f4f4f49b.obj",
        mini_dataset_folder_path + "02691156/ce337df2f75801eeb07412c80bd835.obj",
    ]
    file1_list = os.listdir(mini_dataset_folder_path + '03001627/')
    file2_list = os.listdir(mini_dataset_folder_path + '02691156/')

    mesh_file_path_list = []

    add_num = 0
    for file1 in file1_list:
        if file1[-4:] != '.obj':
            continue

        mesh_file_path_list.append(mini_dataset_folder_path + '03001627/' + file1)
        add_num += 1
        if add_num >= 10:
            break

    add_num = 0
    for file2 in file2_list:
        if file2[-4:] != '.obj':
            continue

        mesh_file_path_list.append(mini_dataset_folder_path + '02691156/' + file2)
        add_num += 1
        if add_num >= 10:
            break


    sample_point_num = 400
    rad_density = 5

    asdf_autoencoder_sampler = ASDFAutoEncoderSampler(model_file_path, device)
    asdf_autoencoder_sampler.sample(
        mesh_file_path_list, sample_point_num, rad_density)
    return True
