from Dataset import RoadNetworkDataset, DEVICE
from TrainEvalTest.Experiment import Experiment
from Models import *
from JimmyTorch.DiffusionModels import DDIM
import pandas as pd

def train():
    experiment = Experiment("PRedict epsilon")

    experiment.train_set_cfg.folder_path = "Dataset/Tokyo"
    experiment.train_set_cfg.batch_size = 32
    experiment.eval_set_cfg.batch_size = 32
    experiment.eval_set_cfg.folder_path = "Dataset/Tokyo"
    experiment.test_set_cfg.folder_path = "Dataset/Tokyo"
    dummy_dataset = experiment.test_set_cfg.build()

    vae = W2GVAE(
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
    ).to(DEVICE)
    vae.loadFrom("Runs/Tokyo_Dataset/W2GVAE/251027_120831/last.pth")
    vae.eval()

    ddm = DDIM(
        min_beta=0.0001,
        max_beta=0.05,
        max_diffusion_step=500,
        device=DEVICE,
        scale_mode="quadratic",
        skip_step=5
    )

    experiment.model_cfg.cls = T2WDiT
    experiment.model_cfg.add(
        ddm=ddm,
        vae=vae,
        d_time=256,
        n_heads=8,
        depth_factor=2,     # 1 for small model, 2 for normal model, 3 for large model
        width_factor=2,
        dropout=0.1,
        pred_target="epsilon"
    )

    experiment.optimizer_cfg.lr = 2e-4
    experiment.model_cfg.compile_model = True
    experiment.model_cfg.mixed_precision = True
    experiment.model_cfg.clip_grad = 1.0
    experiment.constants["eval_interval"] = 10
    experiment.constants["n_epochs"] = 2000
    experiment.constants["checkpoint_interval"] = 500
    experiment.lr_scheduler_cfg.patience = 50

    trainer = experiment.start()

    return trainer

def test(vae_weight_path: str,
         dit_weight_path: str,
         dataset_folder: str,
         noise_level: float,
         n_trajs: int,
         pred_target: str = "v",
         kl_weight: float = 1e-6,
         ddm_skip_step: int = 10,
         vae_threshold: float = 0.5) -> pd.DataFrame:
    experiment = Experiment("T2WDiT test")
    experiment.test_set_cfg.batch_size = 100
    experiment.test_set_cfg.folder_path = dataset_folder
    experiment.test_set_cfg.traj_noise_std = noise_level
    experiment.test_set_cfg.add(truncate_n_trajs=n_trajs)
    test_set = experiment.test_set_cfg.build()

    vae = W2GVAE(
        walks_shape=[test_set.N_trajs_per_sample, test_set.L_walk, test_set.N_interp, 2],
        d_encode=32,
        n_heads=8,
        depth_factor=1,     # 1 for small model, 2 for normal model, 3 for large model
        width_factor=1,
        dropout=0.0,
        bce_warmup=2000,
        kl_warmup=2000,
        kl_weight=kl_weight,
        kl_free_bits=0.1,
        threshold=vae_threshold,
    ).to(DEVICE)
    vae.loadFrom(vae_weight_path)
    vae.eval()

    ddm = DDIM(
        min_beta=0.0001,
        max_beta=0.05,
        max_diffusion_step=500,
        device=DEVICE,
        scale_mode="quadratic",
        skip_step=ddm_skip_step
    )

    DiT = T2WDiT(
        ddm=ddm,
        vae=vae,
        d_time=256,
        n_heads=8,
        depth_factor=2,     # 1 for small model, 2 for normal model, 3 for large model
        width_factor=2,
        dropout=0.1,
        pred_target=pred_target
    ).to(DEVICE)
    DiT.loadFrom(dit_weight_path)
    DiT.eval()

    return experiment.test(DiT, test_set, trials=1)


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

    vae = W2GVAE(
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

    ddm = DDIM(
        min_beta=0.0001,
        max_beta=0.05,
        max_diffusion_step=500,
        device=DEVICE,
        scale_mode="quadratic",
        skip_step=5
    )

    model = T2WDiT(
        ddm=ddm,
        vae=vae,
        d_time=256,
        n_heads=8,
        depth_factor=2,     # 1 for small model, 2 for normal model, 3 for large model
        width_factor=2,
        dropout=0.1,
        pred_target="epsilon"
    ).to(DEVICE)
    model.eval()

    # Record initial GPU memory
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GiB
    print(f"Initial GPU memory: {initial_memory:.2f} GiB")

    start = time()

    first_data_dict = next(iter(test_set))
    z, _ = vae.encode(first_data_dict["walks"])

    running_memory = []

    for data_dict in tqdm(test_set, total=test_set.n_batches):
        with torch.no_grad():
            t = torch.randint(0, 500, (data_dict["trajs"].shape[0],), device=DEVICE)
            model(z, t, data_dict["trajs"])
        # Record in-loop GPU memory
        peak_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)
        running_memory.append(peak_memory)

    print(f"Memory used during testing (max): {max(running_memory) - initial_memory:.3f} GiB")

    end = time()
    print(f"Per batch time: {(end - start) / test_set.n_batches} seconds")

    calculate_flops(
        model,
        args=[z[:1], t[:1], data_dict["trajs"][:1]],
        print_detailed=False,
        output_precision=4
    )


if __name__ == '__main__':
    train()