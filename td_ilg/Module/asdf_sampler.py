import torch
import numpy as np

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

        id = 0
        categories = torch.Tensor([id]).long()

        print("start model cond")
        cond = model.class_enc(categories)
        print("cond:")
        print(cond.shape)

        asdf_params = model.sample(cond).cpu().numpy()

        print(asdf_params.shape)

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
        render = True
        save_folder_path = "./output/test1/"
        sample_direction_num = 200
        direction_upscale = 4

        rad_density = 100
        cone_render_scale = 0.5
        cone_color = [0, 1, 0]

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
        asdf_model.loadParams(asdf_params[0])
        asdf_model.renderDetectPoints(rad_density)
        asdf_model.renderDetectMaskViewCones(rad_density, cone_render_scale, cone_color)

        cone_boundaries = []
        cone_directions = []
        cone_positions = []
        cone_heights = []
        sample_points_per_anchor = asdf_model.toAnchorDetectPointsList(rad_density)
        for anchor_idx in range(asdf_model.anchorNum()):
            mask_boundary_points = toData(
                asdf_model.toAnchorDetectBoundaryPoints(anchor_idx, rad_density),
                "numpy",
            )
            cone_boundaries.append(mask_boundary_points)
            direction = toData(asdf_model.toAnchorDirection(anchor_idx), "numpy")
            cone_directions.append(direction)
            position = toData(asdf_model.toAnchorPositionParams(anchor_idx), "numpy")
            cone_positions.append(position)
            cone_height = toData(asdf_model.toAnchorHeightVector(anchor_idx), "numpy")
            cone_heights.append(cone_height)

        renderPoints(np.vstack(cone_boundaries), "cone_boundaries")
        renderPoints(np.vstack(sample_points_per_anchor), "sample_points_per_anchor")
        return True
