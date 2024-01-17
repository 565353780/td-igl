import torch
# import torch.backends.cudnn as cudnn

from td_ilg.Model.class_encoder import ClassEncoder

# cudnn.benchmark = True


class Sampler(object):
    def __init__(self) -> None:
        self.model_path = "./test.pth"
        self.device = "cpu"
        self.category = 0
        return

    @torch.no_grad()
    def sample(self) -> bool:
        """
        model = ClassEncoder(
            ninp=1024,
            nhead=16,
            nlayers=24,
            nclasses=55,
            coord_vocab_size=256,
            latent_vocab_size=1024,
            reso=512,
        )
        """
        model = ClassEncoder(
            ninp=16,
            nhead=2,
            nlayers=24,
            nclasses=55,
            coord_vocab_size=256,
            latent_vocab_size=1024,
            reso=12,
        )

        model.to(self.device)
        # checkpoint = torch.load(self.model_pth, map_location="cpu")
        # model.load_state_dict(checkpoint["model"])
        model.eval()

        id = self.category
        categories = torch.Tensor([id]).long()

        print("start model cond")
        cond = model.class_enc(categories)
        print("cond:")
        print(cond.shape)

        x, y, z, latent = model.sample(cond)

        print(x.shape)
        print(y.shape)
        print(z.shape)
        print(latent.shape)
        return True
