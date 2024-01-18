import sys

sys.path.append("../data-convert/")
sys.path.append("../spherical-harmonics/")
sys.path.append("../a-sdf/")

def demo():
    from td_ilg.Module.asdf_autoencoder_trainer import ASDFAutoEncoderTrainer

    print_progress = True

    asdf_autoencoder_trainer = ASDFAutoEncoderTrainer()
    asdf_autoencoder_trainer.loadSummaryWriter()
    asdf_autoencoder_trainer.train(print_progress)
    return True
