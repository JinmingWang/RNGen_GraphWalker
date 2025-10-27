from .components import *
from .loss import *
from ..Metrics import *
import matplotlib
matplotlib.use("Agg")   # Use a non-interactive backend for running in terminal
import matplotlib.pyplot as plt
import contextlib


class W2GVAE(JimmyModel):
    def __init__(self,
                 walks_shape: List[int],
                 d_encode: int,
                 n_heads: int = 8,
                 depth_factor: int = 2,
                 width_factor: int = 2,
                 dropout: float = 0.0,
                 bce_warmup: int = 1000,
                 kl_warmup: int = 1000,
                 kl_weight: float = 1e-3,
                 kl_free_bits: float = 0.1,
                 threshold: float=0.5,
                 **JM_kwargs):
        super().__init__(**JM_kwargs)
        self.N_walks, self.L_walk, self.N_interp, _ = walks_shape
        self.d_enc = d_encode
        self.n_heads = n_heads
        self.df = depth_factor
        self.wf = width_factor

        # Input (B, N_trajs, L_walk, N_interp, 2)
        self.encoder = nn.Sequential(
            Rearrange("B N_walks L_walk N_interp D -> (B N_walks) (N_interp D) L_walk"),
            Conv1DInSiLU(self.N_interp * 2, self.wf * 64, 3, 1, 1),
            *[Res1D(self.wf * 64, 4) for _ in range(2 * self.df)],
            Rearrange("(B N_walks) D L_walk -> B (N_walks L_walk) D", N_walks=self.N_walks),
            nn.Linear(self.wf * 64, self.wf * 96),
            *[VAEBlock(self.wf * 96, self.wf * 192, n_heads, self.N_walks, dropout) for _ in range(2 * self.df)],
            Rearrange("B (N_walks L_walk) D -> B N_walks L_walk D", N_walks=self.N_walks),
            MLP([self.wf * 96, self.wf * 96, self.wf * 96, d_encode * 2], act=nn.SiLU(inplace=True))
        )
        nn.init.zeros_(self.encoder[-1][-1].weight)
        nn.init.zeros_(self.encoder[-1][-1].bias)
        self.encoder_shortcut = nn.Sequential(
            Rearrange("B N_walks L_walk N_interp D -> B N_walks L_walk (N_interp D)"),
            nn.Linear(self.N_interp * 2, d_encode),
        )

        self.decoder_shared = nn.Sequential(
            Rearrange("B N_walks L_walk D -> (B N_walks) D L_walk"),
            Conv1DInSiLU(d_encode, self.wf * 64, 3, 1, 1),
            *[Res1D(self.wf * 64, 4) for _ in range(2 * self.df)],
            Rearrange("(B N_walks) D L_walk -> B (N_walks L_walk) D", N_walks=self.N_walks),
            nn.Linear(self.wf * 64, self.wf * 96),
            nn.SiLU(inplace=True),
            nn.Linear(self.wf * 96, self.wf * 192),
            *[VAEBlock(self.wf * 192, self.wf * 384, n_heads, self.N_walks, dropout) for _ in range(4 * self.df)],
            nn.Linear(self.wf * 192, self.wf * 192),
        )
        nn.init.zeros_(self.decoder_shared[-1].weight)
        nn.init.zeros_(self.decoder_shared[-1].bias)
        self.decoder_shortcut = nn.Sequential(
            Rearrange("B N_walks L_walk D -> B (N_walks L_walk) D", N_walks=self.N_walks),
            nn.Linear(d_encode, self.wf * 192),
        )

        self.edges_head = MLP([self.wf * 192, self.wf * 128, self.wf * 128, self.N_interp * 4], act=nn.SiLU(inplace=True))

        self.deduplicator = Deduplicator(self.wf * 192, 32 * self.wf, threshold)

        self.train_loss_names = ["Train/Main", "Train/BCE", "Train/MSE", "Train/KLD", "Train/z_var"]
        self.eval_loss_names = ["Eval/Main", "Eval/BCE", "Eval/MSE", "Eval/KLD", "Eval/Hungarian_MSE", "Eval/Hungarian_MAE", "Eval/Chamfer_MSE", "Eval/Chamfer_MAE", "Eval/EdgeLength"]
        self.bce_func = LowerTriangularBCELoss()
        self.mse_func = EdgeMSELoss()
        self.warmup_steps = 0
        self.bce_warmup = bce_warmup
        self.kl_warmup = kl_warmup
        self.kl_weight = kl_weight
        self.kld_func = KLDLoss(free_bits=kl_free_bits)


    def encode(self, walks):
        z_mean, z_logvar = torch.split(self.encoder(walks), self.d_enc, dim=-1)
        z_mean = z_mean + self.encoder_shortcut(walks)
        # For numerical stability, clamp the logvar
        z_logvar = 10.0 * torch.tanh(z_logvar / 10.0)
        return z_mean, z_logvar


    def decode(self, z):
        # z: (B, N_walks, L_walk, d_enc)
        x = self.decoder_shared(z) + self.decoder_shortcut(z)  # (B, N_edges, 384))
        duplicate_edges = self.edges_head(x)[..., :self.N_interp * 2]  # (B, N_edges, N_interp * 2)
        sim_mat = self.deduplicator(duplicate_edges.detach(), x)
        return duplicate_edges.unflatten(-1, (-1, 2)), sim_mat


    @staticmethod
    def reparameterize(z_mean, z_logvar):
        epsilon = torch.randn_like(z_mean)
        z = z_mean + torch.exp(0.5 * z_logvar) * epsilon
        return z


    def forward(self, walks):
        z_mean, z_logvar = self.encode(walks)
        z = self.reparameterize(z_mean, z_logvar)
        duplicate_edges, affinity_mat = self.decode(z)
        return z_mean, z_logvar, duplicate_edges, affinity_mat


    def getGraphs(self, duplicate_edges, affinity_mat):
        return self.deduplicator.getGraphs(duplicate_edges, affinity_mat)


    def trainStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        if self.warmup_steps < self.kl_warmup:
            self.warmup_steps += 1
        kl_weight = self.kl_weight * self.warmup_steps / self.kl_warmup
        bce_weight = self.warmup_steps / self.bce_warmup

        with torch.autocast("cuda", torch.float16) if self.mixed_precision else contextlib.nullcontext():
            z_mean, z_logvar, duplicate_edges, affinity_mat = self(data_dict["walks"])
            loss_kld = self.kld_func(z_mean, z_logvar) * kl_weight

            target_edges = data_dict["walks"].flatten(1, 2)     # (B, N_walks, L_walk, N_interp, 2) -> (B, N_edges, N_interp, 2)
            target_points = target_edges.flatten(1, 2)  # (B, N_edges, N_interp, 2) -> (B, N_points, 2)
            target_point_wise_dist = torch.cdist(target_points, target_points, p=2)  # (B, N_points, N_points)
            pred_points = duplicate_edges.flatten(1, 2)  # (B, N_edges, N_interp, 2) -> (B, N_points, 2)
            pred_point_wise_dist = torch.cdist(pred_points, pred_points, p=2)  # (B, N_points, N_points)

            loss_mse = self.mse_func(duplicate_edges, target_edges) + 0.5 * self.mse_func(pred_point_wise_dist, target_point_wise_dist)
            target_mat, target_mask = getTargetUniquenessMask(target_edges, data_dict["graphs"])
            loss_bce = self.bce_func(affinity_mat, target_mat) * bce_weight
            loss = loss_bce + loss_mse + loss_kld

        self.backwardOptimize(loss)

        return ({"Train/Main": loss.item(), "Train/BCE": loss_bce.item(), "Train/MSE": loss_mse.item(), "Train/KLD": loss_kld.item(),
                    "Train/z_var": z_logvar.mean().item()},
                {"output": duplicate_edges.detach()})


    def evalStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        B = data_dict['trajs'].shape[0]
        with torch.no_grad():
            z_mean, z_logvar, duplicate_edges, affinity_mat = self(data_dict["walks"])
            loss_kld = self.kld_func(z_mean, z_logvar) * self.kl_weight
            loss_mse = self.mse_func(duplicate_edges, data_dict["walks"].flatten(1, 2))
            target_mat, target_mask = getTargetUniquenessMask(data_dict["walks"].flatten(1, 2),
                                                              data_dict["graphs"])
            loss_bce = self.bce_func(affinity_mat, target_mat)
            loss_total = loss_bce + loss_mse + loss_kld
            graphs = self.getGraphs(duplicate_edges, affinity_mat)  # B * (N_edges, N_interp, 2)
            unpadded_target_graphs = [data_dict["graphs"][b][:data_dict["N_edges_per_graph"][b]] for b in range(B)]
            hungarian_mae, hungarian_mse = hungarianMetric(graphs, unpadded_target_graphs)
            chamfer_mae, chamfer_mse = chamferMetric(graphs, unpadded_target_graphs)
            edge_length = edgeLengthMetric(graphs, unpadded_target_graphs)

            duplicate_edges_no_noise, sim_mat_no_noise = self.decode(z_mean[0:1])  # decode without noise
            graph_no_noise = self.getGraphs(duplicate_edges_no_noise, sim_mat_no_noise)[0]

        plt.close("all")
        fig, ax = plt.subplots(1, 5, figsize=(15, 3))
        ax[0].set_title("Input / GT walks")
        ax[0].axis('equal')
        for edge in  data_dict["walks"][0].flatten(0, 1).cpu().numpy():
            # edge: (N_interp, 2), plt a curve with ending points
            ax[0].plot(edge[:, 0], edge[:, 1], color='blue', alpha=0.3)
            ax[0].scatter(edge[[0, -1], 0], edge[[0, -1], 1], color='blue', s=10)

        ax[1].set_title("Output walks")
        ax[1].axis('equal')
        for edge in duplicate_edges[0].cpu().numpy():
            # edge: (N_interp, 2), plt a curve with ending points
            ax[1].plot(edge[:, 0], edge[:, 1], color='blue', alpha=0.3)
            ax[1].scatter(edge[[0, -1], 0], edge[[0, -1], 1], color='blue', s=10)

        ax[2].set_title("Graph")
        ax[2].axis('equal')
        for edge in graphs[0].cpu().numpy():
            # edge: (N_interp, 2), plt a curve with ending points
            ax[2].plot(edge[:, 0], edge[:, 1], color='red', alpha=0.3)
            ax[2].scatter(edge[[0, -1], 0], edge[[0, -1], 1], color='red', s=10)

        ax[3].set_title("Output walks (no noise)")
        ax[3].axis('equal')
        for edge in duplicate_edges_no_noise[0].cpu().numpy():
            # edge: (N_interp, 2), plt a curve with ending points
            ax[3].plot(edge[:, 0], edge[:, 1], color='blue', alpha=0.3)
            ax[3].scatter(edge[[0, -1], 0], edge[[0, -1], 1], color='blue', s=10)

        ax[4].set_title("Graph (no noise)")
        ax[4].axis('equal')
        for edge in graph_no_noise.cpu().numpy():
            # edge: (N_interp, 2), plt a curve with ending points
            ax[4].plot(edge[:, 0], edge[:, 1], color='red', alpha=0.3)
            ax[4].scatter(edge[[0, -1], 0], edge[[0, -1], 1], color='red', s=10)

        plt.tight_layout()

        return ({"Eval/Main": loss_total.item(), "Eval/BCE": loss_bce.item(), "Eval/MSE": loss_mse.item(), "Eval/KLD": loss_kld.item(),
                    "Eval/Hungarian_MSE": hungarian_mse, "Eval/Hungarian_MAE": hungarian_mae,
                    "Eval/Chamfer_MSE": chamfer_mse, "Eval/Chamfer_MAE": chamfer_mae,
                    "Eval/EdgeLength": edge_length,
                    "Eval/z_var": z_logvar.mean().item()},
                {"fig": fig, "output": z_logvar.detach()})


if __name__ == '__main__':
    from calflops import calculate_flops

    model = W2GVAE([48, 8, 8, 2], 16, 8,
                   2, 2, 0.5,
                   ).cuda()
    # model = model.encoder

    arg1 = torch.rand(1, 48, 8, 8, 2).cuda()

    calculate_flops(
        model,
        args=[arg1],
        print_detailed=True,
        output_precision=4
    )