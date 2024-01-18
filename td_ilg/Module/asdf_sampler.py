import torch
# import torch.backends.cudnn as cudnn

from td_ilg.Model.asdf_class_encoder import ASDFClassEncoder

# cudnn.benchmark = True


class ASDFSampler(object):
    def __init__(self) -> None:
        self.model_path = (
            "/Users/fufu/Nutstore Files/paper-materials-ASDF/Model/checkpoint-63999.pth"
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

        asdf_params = model.sample(cond)

        print(asdf_params.shape)
        return True
