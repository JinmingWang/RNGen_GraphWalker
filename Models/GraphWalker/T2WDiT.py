from torch.nn import Sequential

from .components import *
from .loss import *
from ..Metrics import *
import matplotlib
matplotlib.use("Agg")   # Use a non-interactive backend for running in terminal
import matplotlib.pyplot as plt
import contextlib

class T2WDiT(JimmyModel):
    def __init__(self,
                 ddm: Any,
                 vae: nn.Module,
                 d_time: int,
                 n_heads: int = 8,
                 depth_factor: int = 2,
                 width_factor: int = 2,
                 dropout: float = 0.0,
                 pred_target: Literal["x0", "epsilon", "v"] = "v",
                 **JM_kwargs):
        super().__init__(**JM_kwargs)
        self.N_walks, self.L_walk, self.N_interp = vae.N_walks, vae.L_walk, vae.N_interp
        self.ddm = ddm
        self.frozen_vae = [vae]
        self.d_enc = vae.d_enc
        self.d_time = d_time
        self.n_heads = n_heads
        self.df = depth_factor
        self.wf = width_factor
        self.pred_target = pred_target

        self.t_proj = nn.Sequential(
            nn.Embedding(ddm.T, d_time),
            nn.Linear(d_time, d_time),
            nn.SiLU(inplace=True),
            nn.Linear(d_time, d_time),
            nn.SiLU(inplace=True),
            nn.Unflatten(-1, (1, -1))    # (B, 1, d_time)
        )

        # Input (B, N_trajs, L_traj, 2)
        self.traj_conv_proj = nn.Sequential(
            Rearrange("B N_trajs L_traj xy -> (B N_trajs) xy L_traj"),
            nn.Conv1d(2, self.wf * 24, 3, 1, 1),
            GnSiLUConv1D(self.wf * 24, self.wf * 48, 3, 1, 1, gn_groups=16),
            *[Res1D(self.wf * 48, 1) for _ in range(2 * self.df)],
            nn.Conv1d(self.wf * 48, self.wf * 96, 3, 2, 1),  # downsample by 2
            *[Res1D(self.wf * 96, 1) for _ in range(2 * self.df)],
            nn.AdaptiveAvgPool1d(self.L_walk),
            Rearrange("(B N_trajs) D L_walk -> B (N_trajs L_walk) D", N_trajs=self.N_walks),
        )

        self.traj_attn_proj = nn.ModuleList([
            DiTBlock(self.wf * 96, d_time, self.wf * 192, n_heads, self.N_walks, dropout) for _ in range(4 * self.df)
        ])

        # (B, N_walks, L_walk, d_enc)
        self.noise_conv_proj = nn.Sequential(
            Rearrange("B N_walks L_walk D -> (B N_walks) D L_walk", N_walks=self.N_walks),
            nn.Conv1d(self.d_enc, self.wf * 96, 3, 1, 1),
            *[Res1D(self.wf * 96, 1) for _ in range(self.df)],
            Rearrange("(B N_walks) D L_walk -> B (N_walks L_walk) D", N_walks=self.N_walks),
        )

        self.joint_attn_proj = nn.ModuleList([
            DiTBlock(self.wf * 192, d_time, self.wf * 192, n_heads, self.N_walks, dropout) for _ in range(4 * self.df)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.wf * 192, self.wf * 96),
            nn.SiLU(inplace=True),
            nn.Linear(self.wf * 96, self.wf * 48),
            nn.SiLU(inplace=True),
            nn.Linear(self.wf * 48, self.d_enc * 2),
            nn.Unflatten(1, (self.N_walks, self.L_walk))
        )
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

        self.train_loss_names = ["Train/Main"]
        self.eval_loss_names = ["Eval/Main", "Eval/Hungarian_MSE", "Eval/Hungarian_MAE", "Eval/Chamfer_MSE", "Eval/Chamfer_MAE", "Eval/EdgeLength"]
        self.mse_func = nn.MSELoss()

    @property
    def vae(self):
        return self.frozen_vae[0]

    def forward(self, noise, t, trajs):
        # noise: (B, N_walks, L_walk, d_enc)
        t_emb = self.t_proj(t)  # (B, 1, d_time)
        trajs_emb = self.traj_conv_proj(trajs)  # (B, N_edges, wf*128)
        for block in self.traj_attn_proj:
            trajs_emb = block(trajs_emb, t_emb)

        noise_emb = self.noise_conv_proj(noise)  # (B, N_edges, wf*128)

        x = torch.cat([trajs_emb, noise_emb], dim=-1)  # (B, N_edges, wf*256)
        for block in self.joint_attn_proj:
            x = block(x, t_emb)

        pred = self.head(x)[..., :self.d_enc]  # (B, N_edges, d_enc)

        return pred

    def trainStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        device = data_dict["trajs"].device
        B = data_dict['trajs'].shape[0]
        t = torch.randint(0, self.ddm.T, (B,), device=device).long()    # (B,)
        # data_dict["trajs"]: (B, N_trajs, L_traj, 2)

        with torch.no_grad():
            z_mean, z_logvar = self.vae.encode(data_dict["walks"])
        noise = torch.randn_like(z_mean, device=device)
        z_t = self.ddm.diffuse(z_mean, t, noise)    # (B, N_walks, L_walk, d_enc)

        with torch.autocast("cuda", torch.float16) if self.mixed_precision else contextlib.nullcontext():
            if self.pred_target == "v":
                v_t = self.ddm.computeVelocity(z_mean, noise, t)  # (B, N_walks, L_walk, d_enc)
                pred = self(z_t, t, data_dict["trajs"])
                loss = self.mse_func(pred, v_t)
            elif self.pred_target == "x0":
                pred = self(z_t, t, data_dict["trajs"])
                loss = self.mse_func(pred, z_mean)
            else:
                pred = self(z_t, t, data_dict["trajs"])
                loss = self.mse_func(pred, noise)

        self.backwardOptimize(loss)

        return {"Train/Main": loss.item()}, {"output": pred.detach()}


    def evalStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        B = data_dict['trajs'].shape[0]

        def pred_func(z_t, t):
            pred = self(z_t, t, data_dict["trajs"])
            if self.pred_target == "x0":
                return pred, None, None
            elif self.pred_target == "epsilon":
                return None, pred, None
            else:
                return None, None, pred

        with torch.no_grad():
            z_mean, z_logvar = self.vae.encode(data_dict["walks"])
            noise = torch.randn_like(z_mean, device=z_mean.device)

            pred_z = self.ddm.denoise(noise, pred_func)

            loss = self.mse_func(pred_z, z_mean)

            duplicate_edges, affinity_mat = self.vae.decode(pred_z)

            graphs = self.vae.getGraphs(duplicate_edges, affinity_mat)  # B * (N_edges, N_interp, 2)
            unpadded_target_graphs = [data_dict["graphs"][b][:data_dict["N_edges_per_graph"][b]] for b in range(B)]
            hungarian_mae, hungarian_mse = hungarianMetric(graphs, unpadded_target_graphs)
            chamfer_mae, chamfer_mse = chamferMetric(graphs, unpadded_target_graphs)
            edge_length = edgeLengthMetric(graphs, unpadded_target_graphs)

        plt.close("all")
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        ax[0].set_title("GT walks")
        ax[0].axis('equal')
        for edge in  data_dict["walks"][0].flatten(0, 1).cpu().numpy():
            # edge: (N_interp, 2), plt a curve with ending points
            ax[0].plot(edge[:, 0], edge[:, 1], color='blue', alpha=0.3)
            ax[0].scatter(edge[[0, -1], 0], edge[[0, -1], 1], color='blue', s=10)

        ax[1].set_title("Input trajs")
        ax[1].axis('equal')
        for ti, traj in enumerate(data_dict["trajs"][0].cpu().numpy()):
            ax[1].plot(traj[:, 0], traj[:, 1], color='green', alpha=0.3)

        ax[2].set_title("Output walks")
        ax[2].axis('equal')
        for edge in duplicate_edges[0].cpu().numpy():
            # edge: (N_interp, 2), plt a curve with ending points
            ax[2].plot(edge[:, 0], edge[:, 1], color='blue', alpha=0.3)
            ax[2].scatter(edge[[0, -1], 0], edge[[0, -1], 1], color='blue', s=10)

        ax[3].set_title("Graph")
        ax[3].axis('equal')
        for edge in graphs[0].cpu().numpy():
            # edge: (N_interp, 2), plt a curve with ending points
            ax[3].plot(edge[:, 0], edge[:, 1], color='red', alpha=0.3)
            ax[3].scatter(edge[[0, -1], 0], edge[[0, -1], 1], color='red', s=10)

        plt.tight_layout()

        return ({"Eval/Main": loss,
                 "Eval/Hungarian_MSE": hungarian_mse,
                 "Eval/Hungarian_MAE": hungarian_mae,
                 "Eval/Chamfer_MSE": chamfer_mse,
                 "Eval/Chamfer_MAE": chamfer_mae,
                 "Eval/EdgeLength": edge_length},
                {"fig": fig, "output": [g.detach() for g in graphs]})


    def testStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        B = data_dict['trajs'].shape[0]

        def pred_func(z_t, t):
            pred = self(z_t, t, data_dict["trajs"])
            if self.pred_target == "x0":
                return pred, None, None
            elif self.pred_target == "epsilon":
                return None, pred, None
            else:
                return None, None, pred

        with torch.no_grad():
            z_mean, z_logvar = self.vae.encode(data_dict["walks"])
            noise = torch.randn_like(z_mean, device=z_mean.device)

            pred_z = self.ddm.denoise(noise, pred_func)

            loss = func.mse_loss(pred_z, z_mean, reduction="none").flatten(1).mean(dim=1)

            duplicate_edges, affinity_mat = self.vae.decode(pred_z)

            graphs = self.vae.getGraphs(duplicate_edges, affinity_mat)  # B * (N_edges, N_interp, 2)
            unpadded_target_graphs = [data_dict["graphs"][b][:data_dict["N_edges_per_graph"][b]] for b in range(B)]
            hungarian_mae, hungarian_mse = hungarianMetric(graphs, unpadded_target_graphs, reduction="none")
            chamfer_mae, chamfer_mse = chamferMetric(graphs, unpadded_target_graphs, reduction="none")
            edge_length = edgeLengthMetric(graphs, unpadded_target_graphs, reduction="none")

        return ({"Eval/Main": loss,
                 "Eval/Hungarian_MSE": hungarian_mse,
                 "Eval/Hungarian_MAE": hungarian_mae,
                 "Eval/Chamfer_MSE": chamfer_mse,
                 "Eval/Chamfer_MAE": chamfer_mae,
                 "Eval/EdgeLength": edge_length},
                {"output": [g.detach() for g in graphs]})

if __name__ == '__main__':
    from calflops import calculate_flops
    dummy_vae = type('obj', (object,), {'N_walks':8, 'L_walk':8, 'N_interp':2, 'd_enc': 16})()
    dummy_ddm = type('obj', (object,), {'T':500})()

    model = T2WDiT(dummy_vae, dummy_ddm, 64, 8, 2, 2, 0.0).cuda()

    noise = torch.rand(1, 384, 16).cuda()
    t = torch.randint(0, 500, (1,)).long().cuda()
    trajs = torch.rand(1, 48, 16, 2).cuda()

    calculate_flops(
        model,
        args=[noise, t, trajs],
        print_detailed=True,
        output_precision=4
    )