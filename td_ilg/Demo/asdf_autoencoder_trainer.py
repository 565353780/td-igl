import sys

sys.path.append("../data-convert/")
sys.path.append("../spherical-harmonics/")
sys.path.append("../a-sdf/")

def demo():
    from td_ilg.Module.asdf_autoencoder_trainer import ASDFAutoEncoderTrainer

    model_file_path = './output/poly-lr_1e-2_16batch/model_best.pth'
    print_progress = True

    asdf_autoencoder_trainer = ASDFAutoEncoderTrainer()
    asdf_autoencoder_trainer.loadSummaryWriter()
    #asdf_autoencoder_trainer.loadModel(model_file_path, True)
    asdf_autoencoder_trainer.train(print_progress)
    return True
