from torch.optim.lr_scheduler import ReduceLROnPlateau

from TrainEvalTest.Trainer import Trainer
from Dataset import RoadNetworkDataset
from JimmyTorch.Datasets import DEVICE
from JimmyTorch.Training import *
from JimmyTorch.Models import JimmyModel
from JimmyTorch.DynamicConfig import DynamicConfig
import torch
from typing import *
from datetime import datetime
import os
from rich import print as rprint
import pandas as pd
import numpy as np
from tqdm import tqdm


class Experiment:
    """
    This is an example of an experiment class that defines the hyperparameters and constants for the experiment.
    For other type of experiments, or your customized trainer, you should write a new experiment class to accommodate
    the new set of hyperparameters and constants.
    """

    def __init__(self, comments: str):
        self.comments = comments

        self.optimizer_cfg = DynamicConfig(torch.optim.AdamW,
                                           lr=2e-4,
                                           amsgrad=True)

        self.model_cfg = DynamicConfig(JimmyModel,
                                       mixed_precision=True,
                                       compile_model=True,
                                       clip_grad=0.0)

        self.train_set_cfg = DynamicConfig(RoadNetworkDataset,
                                           folder_path="Dataset/Tokyo",
                                           batch_size=32,
                                           drop_last=True,
                                           shuffle=True,
                                           img_H=1,
                                           img_W=1,
                                           permute_seq=True,
                                           enable_aug=False,
                                           need_image=False,
                                           need_heatmap=False,
                                           need_nodes=False,
                                           traj_noise_std=0.015,
                                           set_name="train",
                                           )

        self.eval_set_cfg = DynamicConfig(RoadNetworkDataset,
                                           folder_path="Dataset/Tokyo",
                                           batch_size=32,
                                           drop_last=True,
                                           shuffle=True,
                                           img_H=1,
                                           img_W=1,
                                           permute_seq=False,
                                           enable_aug=False,
                                           need_image=False,
                                           need_heatmap=False,
                                           need_nodes=False,
                                           traj_noise_std=0.015,
                                           set_name="eval",
                                           )

        self.test_set_cfg = DynamicConfig(RoadNetworkDataset,
                                           folder_path="Dataset/Tokyo",
                                           batch_size=32,
                                           drop_last=True,
                                           shuffle=False,
                                           img_H=1,
                                           img_W=1,
                                           permute_seq=False,
                                           enable_aug=False,
                                           need_image=False,
                                           need_heatmap=False,
                                           need_nodes=False,
                                           traj_noise_std=0.015,
                                           set_name="test",
                                           )


        # The default hyperparameters for the experiment.
        self.lr_scheduler_cfg = DynamicConfig(ReduceLROnPlateau,
                                            mode="min",
                                            factor=0.5,
                                            patience=20,
                                            threshold=1e-7,
                                            min_lr=1e-6,
                                            verbose=False)

        # Other constants for the experiment.
        self.constants = {
            "n_epochs": 2000,
            "moving_avg": 1000,
            "eval_interval": 5,
            "random_seed": 3407,
            "checkpoint_interval": 500,
        }

        self.trainer_type = Trainer


    def __str__(self):
        return (f"Experiment{{\n"
                f"\ttrainset={self.train_set_cfg}\n"
                f"\tevalset={self.eval_set_cfg}\n"
                f"\ttestset={self.test_set_cfg}\n"
                f"\tmodel={self.model_cfg}\n"
                f"\tlr_scheduler={self.lr_scheduler_cfg}\n"
                f"\tconstants={self.constants}\n}}")


    def __repr__(self):
        return self.__str__()


    def start(self, checkpoint: str = None) -> Trainer:
        """
        Start the experiment with the given comments.
        :param comments: Comments to be added to the Experiment.
        :return: A `JimmyTrainer` object with almost everything during a training session.
        """
        rprint(f"[#00ff00]--- Start Experiment \"{self.comments}\" ---[/#00ff00]")

        torch.manual_seed(self.constants["random_seed"])
        torch.cuda.manual_seed(self.constants["random_seed"])
        del self.constants["random_seed"]

        train_set = self.train_set_cfg.build()
        eval_set = self.eval_set_cfg.build()

        self.model_cfg.optimizer_cls = self.optimizer_cfg.cls
        self.model_cfg.optimizer_args = {"lr": self.optimizer_cfg.lr,
                                         "amsgrad": self.optimizer_cfg.amsgrad}
        model = self.model_cfg.build().to(DEVICE)
        model.initialize()

        if checkpoint is not None:
            model.loadFrom(checkpoint)

        self.lr_scheduler_cfg.optimizer = model.optimizer
        lr_scheduler = self.lr_scheduler_cfg.build()

        trainer_kwargs = {"comments": self.comments, "train_set": train_set, "eval_set": eval_set, "model": model, "lr_scheduler": lr_scheduler}
        trainer_kwargs.update(self.constants)

        # Create Experiment directories
        now_str = datetime.now().strftime("%y%m%d_%H%M%S")
        dataset_name = trainer_kwargs["train_set"].__class__.__name__
        model_name = model.__class__.__name__
        save_dir = f"Runs/{dataset_name}/{model_name}/{now_str}/"
        log_dir = save_dir

        # Create directories if they do not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(os.path.join(log_dir, "model_arch.txt"), "w") as f:
            f.write(str(model))

        with open(os.path.join(log_dir, "comments.txt"), "w") as f:
            f.write(f"{self.comments}\n{self.__str__()}")

        rprint(f"[blue]Save directory: {save_dir}.[/blue]")
        rprint(f"[blue]Log directory: {log_dir}.[/blue]")

        trainer_kwargs["log_dir"] = log_dir
        trainer_kwargs["save_dir"] = save_dir

        trainer = self.trainer_type(**trainer_kwargs)
        trainer.start()

        rprint(f"[blue]Training done. Start testing.[/blue]")
        test_set = self.test_set_cfg.build()
        test_losses = trainer.evaluate(test_set, compute_avg=False)

        test_report = pd.DataFrame.from_dict(test_losses)
        test_report.to_csv(os.path.join(log_dir, "test_report.csv"))

        rprint(f"[blue]Testing done. Reports saved to: {os.path.join(log_dir, 'test_report.csv')}.[/blue]")

        return trainer


    def test(self, model, test_set, trials: int = 1) -> pd.DataFrame:
        rprint(f"[blue]Testing on {self.comments}[/blue]")

        all_trial_losses = []
        B = test_set.batch_size

        for ti in range(trials):
            test_losses = {name: torch.zeros(test_set.n_samples).to(DEVICE) for name in model.eval_loss_names}
            for i, data_dict in enumerate(tqdm(test_set, total=test_set.n_batches, desc=f"Trial {ti+1}/{trials}")):
                loss_dict, output_dict = model.testStep(data_dict)

                for name in model.eval_loss_names:
                    test_losses[name][i*B:(i+1)*B] = loss_dict[name].detach().cpu()

            test_losses["trial"] = np.ones(test_set.n_samples) * ti + 1

            all_trial_losses.append(test_losses)

        overall_losses = {name: torch.cat([trial[name] for trial in all_trial_losses], dim=0).cpu().numpy() for name in model.eval_loss_names}
        overall_losses["trial"] = np.concatenate([trial["trial"] for trial in all_trial_losses], axis=0)

        test_report = pd.DataFrame.from_dict(overall_losses)
        return test_report

        







