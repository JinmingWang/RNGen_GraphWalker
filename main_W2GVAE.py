from Dataset import RoadNetworkDataset, DEVICE
from TrainEvalTest.Experiment import Experiment
from Models import *
import pandas as pd

def train():
    experiment = Experiment("Train with pos enc")

    experiment.train_set_cfg.folder_path = "Dataset/Tokyo"
    experiment.train_set_cfg.batch_size = 32
    experiment.eval_set_cfg.batch_size = 64
    dummy_dataset = experiment.test_set_cfg.build()

    # According to experiment, df = 1, wf = 2 is very economic and its performance is quite good

    experiment.model_cfg.cls = W2GVAE
    experiment.model_cfg.add(
        walks_shape=[dummy_dataset.N_trajs_per_sample, dummy_dataset.L_walk, dummy_dataset.N_interp, 2],
        d_encode=32,
        n_heads=8,
        depth_factor=1,     # 1 for small model, 2 for normal model, 3 for large model
        width_factor=1,
        dropout=0.0,
        bce_warmup=2000,
        kl_warmup=2000,
        kl_weight=1e-6,
        kl_free_bits=0.1,
        threshold=0.5,
    )

    experiment.optimizer_cfg.lr = 2e-4
    experiment.model_cfg.compile_model = True
    experiment.model_cfg.mixed_precision = True
    experiment.model_cfg.clip_grad = 1.0
    experiment.constants["n_epochs"] = 2000
    experiment.lr_scheduler_cfg.patience = 50

    return experiment.start()


def getCost():
    from tqdm import tqdm
    from time import time
    import torch
    from calflops import calculate_flops
    experiment = Experiment("Get Computational Cost")
    experiment.test_set_cfg.folder_path = "Dataset/Tokyo"
    experiment.test_set_cfg.set_name = "test"
    experiment.test_set_cfg.need_image = True
    experiment.test_set_cfg.need_heatmap = True
    experiment.test_set_cfg.need_nodes = True
    experiment.test_set_cfg.img_H = 128
    experiment.test_set_cfg.img_W = 128
    experiment.test_set_cfg.batch_size = 100
    test_set = experiment.test_set_cfg.build()

    model = W2GVAE(
        walks_shape=[test_set.N_trajs_per_sample, test_set.L_walk, test_set.N_interp, 2],
        d_encode=32,
        n_heads=8,
        depth_factor=1,     # 1 for small model, 2 for normal model, 3 for large model
        width_factor=1,
        dropout=0.0,
        bce_warmup=2000,
        kl_warmup=2000,
        kl_weight=1e-6,
        kl_free_bits=0.1,
        threshold=0.5,
    ).to(DEVICE)
    model.eval()

    # Record initial GPU memory
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GiB
    print(f"Initial GPU memory: {initial_memory:.2f} GiB")

    start = time()

    running_memory = []

    for data_dict in tqdm(test_set, total=test_set.n_batches):
        with torch.no_grad():
            model(data_dict["walks"])
        # Record in-loop GPU memory
        peak_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)
        running_memory.append(peak_memory)

    print(f"Memory used during testing (max): {max(running_memory) - initial_memory:.3f} GiB")

    end = time()
    print(f"Per batch time: {(end - start) / test_set.n_batches} seconds")

    calculate_flops(
        model,
        args=[data_dict["walks"][:1]],
        print_detailed=False,
        output_precision=4
    )


if __name__ == '__main__':
    train()