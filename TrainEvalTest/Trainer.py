import torch
import torch.nn as nn
from Dataset import *
from JimmyTorch.Datasets import MultiThreadLoader, DEVICE, plotTraj
from JimmyTorch.Training import *
from JimmyTorch.Models import JimmyModel
from datetime import datetime
import os
from rich import print as rprint
from typing import Dict, Any
import inspect
import matplotlib.pyplot as plt
import yaml


class Trainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 comments: str,
                 train_set: RoadNetworkDataset,
                 eval_set: RoadNetworkDataset,
                 model: JimmyModel,
                 lr_scheduler: JimmyLRScheduler | torch.optim.lr_scheduler._LRScheduler,
                 log_dir: str,
                 save_dir: str,
                 n_epochs: int,
                 moving_avg: int,
                 eval_interval: int,
                 checkpoint_interval: int) -> None:
        """
        Initialize the trainer with a dataset, model, optimizer, and comments.

        :param train_set: A `JimmyDataset` object that provides the training data. It must return a dictionary containing the input data and target labels.
        :param model: A `JimmyModel` object. The model must implement a `forwardBackward` function that returns a dictionary of loss and output, and must have a `loss_names` attribute that lists the keys of the loss dictionary.
        :param optimizer: A PyTorch optimizer used to update the model parameters.
        :param lr_scheduler: A learning rate scheduler. It can be a `JimmyLRScheduler` or a PyTorch learning rate scheduler. The scheduler must implement an `update` method that takes the current loss as an argument.
        :param log_dir: A string specifying the directory where the training logs will be saved.
        :param save_dir: A string specifying the directory where the model checkpoints will be saved.
        :param moving_avg: An integer specifying the window size for calculating the moving average of the loss. Default is 100.
        :param mixed_precision: A boolean indicating whether to use mixed precision training. Default is False.
        :param clip_grad: A float specifying the maximum gradient norm for gradient clipping. Default is 0.0 (no clipping).
        """
        self.comments = comments
        self.train_set = train_set
        self.eval_set = eval_set
        self.model = model

        if not hasattr(lr_scheduler, 'update'):
            # get number of arguments of lr_scheduler.step()
            num_args = len(inspect.signature(lr_scheduler.step).parameters)
            if num_args == 1:
                lr_scheduler.update = lambda metric: lr_scheduler.step()
            else:
                lr_scheduler.update = lambda metric: lr_scheduler.step(metric)

        self.lr_scheduler = lr_scheduler
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.n_epochs = n_epochs
        self.moving_avg = moving_avg
        self.eval_interval = eval_interval
        self.ckpt_interval = checkpoint_interval


    def start(self) -> None:
        """
        Train the model for a specified number of epochs.

        :param epochs: The number of epochs to train the model.
        """
        pm_log_tags = self.model.train_loss_names + ["LR"]
        tm_log_tags = self.model.train_loss_names + self.model.eval_loss_names + ["LR"]
        # Initialize progress manager and tensorboard manager
        pm = ProgressManager(self.train_set.n_batches, self.n_epochs, 5, 2, custom_fields=pm_log_tags)
        tm = TensorBoardManager(self.log_dir, tags=tm_log_tags, value_types=["scalar"] * len(tm_log_tags))
        tm.writer.add_text("Comments", self.comments)
        tm.register("Visualization", "figure")
        ma_losses = {name: MovingAvg(self.moving_avg) for name in self.model.train_loss_names}

        # runtime_param_buffer.yaml is used to dynamically update the running parameters during training
        # The user can just change the value in this file to change the parameter values
        with open(os.path.join(self.log_dir, "runtime_param_buffer.yaml"), "w") as f:
            yaml.dump({"LR": self.model.optimizer.param_groups[0]["lr"]}, f)

        best_loss = float('inf')

        for epoch in range(self.n_epochs):
            # loader = MultiThreadLoader(self.train_set, 3)
            for i, data_dict in enumerate(self.train_set):
                # forward, backward, optimization
                loss_dict, output_dict = self.model.trainStep(data_dict)

                # Compute moving average of losses
                for loss_name in self.model.train_loss_names:
                    ma_losses[loss_name].update(loss_dict[loss_name])
                    loss_dict[loss_name] = ma_losses[loss_name].get()

                # Update progress manager
                pm.update(epoch, i, LR=self.model.lr, **loss_dict)

            # Update tensorboard
            tm.log(pm.overall_progress, LR=self.model.lr, **loss_dict)

            # Update learning rate scheduler
            prev_lr = self.model.lr
            self.lr_scheduler.update(loss_dict["Train/Main"])
            if self.model.lr != prev_lr:
                with open(os.path.join(self.log_dir, "runtime_param_buffer.yaml"), "w") as f:
                    yaml.dump({"LR": self.model.lr}, f)
            with open(os.path.join(self.log_dir, "runtime_param_buffer.yaml"), "r") as f:
                runtime_params = yaml.safe_load(f)
                self.model.optimizer.param_groups[0]["lr"] = float(runtime_params["LR"])

            if epoch % self.eval_interval == 0:
                eval_losses = self.evaluate(self.eval_set, pm=pm, tm=tm)

                tm.log(pm.overall_progress, **eval_losses)

                eval_loss = eval_losses["Eval/Main"]
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.model.saveTo(os.path.join(self.save_dir, "best.pth"))

                if not torch.isnan(torch.tensor(loss_dict["Train/Main"])):
                    self.model.saveTo(os.path.join(self.save_dir, f"last.pth"))

            if epoch % self.ckpt_interval == 0:
                self.model.saveTo(os.path.join(self.save_dir, f"ckpt_epoch{epoch}.pth"))

        pm.close()


    def evaluate(self,
                 dataset: RoadNetworkDataset,
                 compute_avg: bool=True,
                 pm: ProgressManager = None,
                 tm: TensorBoardManager = None) -> Dict[str, Any]:
        n_batches = dataset.n_batches
        eval_losses = {name: torch.zeros(n_batches).to(DEVICE) for name in self.model.eval_loss_names}
        self.model.eval()

        for i, data_dict in enumerate(dataset):
            loss_dict, output_dict = self.model.evalStep(data_dict)

            for name in self.model.eval_loss_names:
                eval_losses[name][i] = loss_dict[name]

        if tm is not None:
            tm.log(pm.overall_progress, Visualization=output_dict["fig"])

        self.model.train()

        if compute_avg:
            return {name: torch.mean(eval_losses[name]).item() for name in self.model.eval_loss_names}

        return {name: eval_losses[name].cpu().numpy() for name in self.model.eval_loss_names}