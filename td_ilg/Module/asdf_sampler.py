import torch
import numpy as np
from tqdm import tqdm

from data_convert.Method.data import toData
from a_sdf.Model.asdf_model import ASDFModel
from a_sdf.Method.render import renderPoints

from td_ilg.Model.asdf_class_encoder import ASDFClassEncoder


class ASDFSampler(object):
    def __init__(self) -> None:
        self.model_path = (
            "/Users/fufu/Nutstore Files/paper-materials-ASDF/Model/checkpoint-71999.pth"
        )
        self.device = "cpu"
        self.resolution = 100
        return

    def toInitialASDFModel(self) -> ASDFModel:
        max_sh_3d_degree = 4
        max_sh_2d_degree = 4
        use_inv = True
        method_name = "torch"
        dtype = torch.float32
        device = "cpu"
        epoch = 10000
        lr = 5e-3
        weight_decay = 1e-4
        factor = 0.8
        patience = 10
        min_lr = lr * 1e-1
        render = False
        save_folder_path = "./output/test1/"
        sample_direction_num = 200
        direction_upscale = 4

        asdf_model = ASDFModel(
            max_sh_3d_degree,
            max_sh_2d_degree,
            use_inv,
            method_name,
            dtype,
            device,
            epoch,
            lr,
            weight_decay,
            factor,
            patience,
            min_lr,
            render,
            save_folder_path,
            sample_direction_num,
            direction_upscale,
        )

        return asdf_model

    @torch.no_grad()
    def sample(self) -> bool:
        """
        model = ASDFClassEncoder(
            ninp=1024,
            nhead=16,
            nlayers=24,
            nclasses=55,
            coord_vocab_size=256,
            latent_vocab_size=1024,
            reso=512,
        )
        """
        model = ASDFClassEncoder(
            asdf_dim=40,
            ninp=16,
            nhead=2,
            nlayers=36,
            nclasses=55,
            coord_vocab_size=256,
            reso=self.resolution,
        )

        model.to(self.device)
        # checkpoint = torch.load(self.model_pth, map_location="cpu")
        # model.load_state_dict(checkpoint["model"])
        model.eval()

        asdf_list = []

        for i in tqdm(range(1)):
            categories = torch.Tensor([0]).long()
            cond = model.class_enc(categories)
            asdf_params = model.sample(cond).cpu().numpy()[0]
            asdf_model = self.toInitialASDFModel()
            asdf_model.loadParams(asdf_params)
            asdf_list.append(asdf_model)

        rad_density = 100

        points = []

        for i in range(len(asdf_list)):
            # asdf_model.renderDetectPoints(rad_density)
            # asdf_model.renderDetectMaskViewCones(rad_density, cone_render_scale, cone_color)

            detect_points = toData(asdf_model.toDetectPoints(rad_density), "numpy")
            detect_points += [0, 0, i * 40]
            points.append(detect_points)

        renderPoints(np.vstack(points), "asdf points")
        return True
