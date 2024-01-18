from td_ilg.Demo.trainer import demo as demo_train
from td_ilg.Demo.asdf_trainer import demo as demo_train_asdf
from td_ilg.Demo.sampler import demo as demo_sample
from td_ilg.Demo.asdf_sampler import demo as demo_sample_asdf

if __name__ == "__main__":
    demo_train()
    demo_train_asdf()
    demo_sample()
    demo_sample_asdf()
