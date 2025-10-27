import torch
from  JimmyTorch.Datasets import *
from tqdm import tqdm
import os
import numpy as np

#In the future, this will have to be added when using torch.load
# torch.serialization.add_safe_globals

def regenerateHeatmaps(all_trajs, traj_lens):
    h, w = 256, 256
    data_count = len(all_trajs)
    N_trajs = all_trajs.shape[1]

    heatmaps = []
    for b in tqdm(range(data_count), desc="Generating heatmaps"):
        trajs = all_trajs[b]
        traj_len = traj_lens[b]
        if torch.all(traj_len != 0):
            points = torch.cat([trajs[i, :traj_len[i]] for i in range(N_trajs)], dim=0)
            min_point = torch.min(points, dim=0, keepdim=True).values
            max_point = torch.max(points, dim=0, keepdim=True).values
            point_range = max_point - min_point

            norm_points = (points - min_point) / point_range

            x_ids = (norm_points[:, 0] * (w - 1)).long()
            y_ids = (norm_points[:, 1] * (h - 1)).long()
            heatmap_flat = torch.zeros(h * w, dtype=torch.float32)
            flat_indices = y_ids * w + x_ids
            heatmap_flat.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32))
            heatmaps.append(heatmap_flat.view(h, w))
        else:
            heatmaps.append(torch.zeros(h, w, dtype=torch.float32))

    return torch.stack(heatmaps, dim=0)


class RoadNetworkDataset(JimmyDataset):
    def __init__(self,
                 folder_path: str,
                 batch_size: int = 32,
                 drop_last: bool = True,
                 shuffle: bool = True,
                 set_name: str = "train",
                 permute_seq: bool = True,
                 enable_aug: bool = False,
                 img_H: int = 256,
                 img_W: int = 256,
                 need_image: bool = False,
                 need_heatmap: bool = False,
                 need_nodes: bool = False,
                 traj_noise_std: float = 0.02,
                 truncate_n_trajs: int = 64) -> None:
        """
        Initialize the dataset, this class loads data from a cache file
        The cache file is created by using LaDeDatasetCacheGenerator class
        :param path: the path to the cache file
        :param max_trajs: the maximum number of trajectories to use
        :param set_name: the name of the set, can be "train", "test", "debug" or "all"
        """
        dataset_name = os.path.basename(folder_path)
        self.__class__.__name__ = f"{dataset_name}_Dataset"
        super().__init__(batch_size, drop_last, shuffle)

        self.set_name = set_name
        self.permute_seq = permute_seq
        self.enable_aug = enable_aug
        self.img_H = img_H
        self.img_W = img_W
        self.need_image = need_image
        self.need_heatmap = need_heatmap
        self.need_nodes = need_nodes
        self.more_noise_std = traj_noise_std
        self.truncate_n_trajs = truncate_n_trajs

        dataset = torch.load(os.path.join(folder_path, "coords.pt"), weights_only=False)

        # (N_data, N_trajs, L_traj, 2)
        self.trajs = dataset["all_trajs"]
        data_count = len(self.trajs)
        slicing = {"train": slice(data_count - 600),
                   "eval": slice(data_count - 600, data_count - 500),
                   "test": slice(data_count - 500, None),
                   "debug": slice(data_count - 200, None),
                   "all": slice(data_count)}[set_name]

        # Data Loading
        self.trajs = self.trajs[slicing][:, :truncate_n_trajs]
        # self.trajs[:, 24:] = 0  # reduce number of trajs provided by half
        # (N_data, N_trajs, L_walk, N_interp, 2)
        self.walks = dataset["all_walks"][slicing][:, :truncate_n_trajs]
        # (N_data, N_edges, N_interp, 2)
        self.graphs = dataset["graphs"][slicing]

        # Get the data dimensions
        self.n_samples = self.trajs.shape[0]
        self.N_trajs_per_sample = self.trajs.shape[1]
        self.L_traj = self.trajs.shape[2]

        self.N_walks_per_sample = self.walks.shape[1]
        self.L_walk = self.walks.shape[2]
        self.N_interp = self.walks.shape[3]

        self.N_pad_edges_per_graph = self.graphs.shape[1]

        self.N_edges_per_graph = dataset["graph_edge_counts"][slicing].to(torch.int32)

        self.bboxes = dataset["graph_bboxes"][slicing]

        if need_image or need_heatmap:
            img_hmap = torch.load(os.path.join(folder_path, "imgs_hmaps.pt"), weights_only=False)
            if need_image:
                # (N_data, 3, H, W)
                self.images = img_hmap["images"][slicing]
                self.images = torch.nn.functional.interpolate(self.images, (img_H, img_W), mode="bilinear")
            if need_heatmap:
                # (N_data, 1, H, W)
                self.heatmaps = img_hmap["heatmaps"][slicing]
                # self.heatmaps = regenerateHeatmaps(self.trajs, dataset["traj_lens"]).unsqueeze(1)    # regenerate heatmaps
                self.heatmaps = torch.nn.functional.interpolate(self.heatmaps, (img_H, img_W), mode="nearest")

        if need_nodes:
            self.edgesToNodesAdj()

        print(str(self))

    def __str__(self):
        return f"RoadNetworkDataset: {self.set_name} set with {self.n_samples} samples packed to {self.n_batches} batches"


    def __repr__(self):
        return self.__str__().replace("\n", ", ")


    def augmentation(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply data augmentation to the given sample
        :return: The augmented sample
        """

        B = batch["trajs"].shape[0]
        point_shift = torch.randn(B, 1, 1, 2).to(DEVICE) * 0.05
        batch["trajs"] += point_shift * (batch["trajs"] != 0)
        batch["walks"] += point_shift.unsqueeze(1) * (batch["walks"] != 0)
        batch["graphs"] += point_shift * (batch["graphs"] != 0)
        if self.need_nodes:
            batch["nodes"] += point_shift.squeeze(1) * (batch["nodes"] != 0)
            batch["edges"] = batch["edges"].unflatten(-1, (-1, 2)) + point_shift.unsqueeze(1)

        if np.random.rand() < 0.5:
            batch["trajs"][..., 0] = -batch["trajs"][..., 0]
            batch["walks"][..., 0] = -batch["walks"][..., 0]
            batch["graphs"][..., 0] = -batch["graphs"][..., 0]
            if self.need_nodes:
                batch["nodes"][..., 0] = -batch["nodes"][..., 0]
                batch["edges"][..., 0] = -batch["edges"][..., 0]

        if np.random.rand() < 0.5:
            batch["trajs"][..., 1] = -batch["trajs"][..., 1]
            batch["walks"][..., 1] = -batch["walks"][..., 1]
            batch["graphs"][..., 1] = -batch["graphs"][..., 1]
            if self.need_nodes:
                batch["nodes"][..., 1] = -batch["nodes"][..., 1]
                batch["edges"][..., 1] = -batch["edges"][..., 1]

        # Random rotate trajs, walks and graphs centered at (0, 0)
        # trajs: (B, N_traj, L_traj, 2)
        # walks: (B, N_traj, L_route, N_interp, 2)
        # graphs: (B, N_edges, N_interp, 2)
        radian = torch.rand(B) * 2 * np.pi
        cos_theta = torch.cos(radian)
        sin_theta = torch.sin(radian)
        rot_matrix = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1).view(B, 2, 2).to(DEVICE)

        batch["trajs"] = torch.einsum("bij,bnlj->bnli", rot_matrix, batch["trajs"])
        batch["walks"] = torch.einsum("bij,bnlkj->bnlki", rot_matrix, batch["walks"])
        batch["graphs"] = torch.einsum("bij,bnlj->bnli", rot_matrix, batch["graphs"])
        if self.need_nodes:
            batch["nodes"] = torch.einsum("bij,bnj->bni", rot_matrix, batch["nodes"])
            batch["edges"] = torch.einsum("bij,bhwnj->bhwni", rot_matrix, batch["edges"]).flatten(-2, -1)

        return batch


    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Return the sample at the given index
        :param idx: the index of the sample
        :return: the sample at the given index
        """
        start = (idx - 1) * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        indices = self._indices[start:end]

        trajs = self.trajs[indices].to(DEVICE)      # (B, N_trajs, L_traj, 2)
        walks = self.walks[indices].to(DEVICE)      # (B, N_trajs, L_route, N_interp, 2)
        graphs = self.graphs[indices].to(DEVICE)    # (B, N_edges, N_interp, 2)

        if self.permute_seq:
            # permute the order of the trajectories and walks
            traj_perm = torch.randperm(trajs.shape[1])
            trajs = trajs[:, traj_perm]
            walks = walks[:, traj_perm]

            # permute the order of the edges
            edges_perm = torch.randperm(graphs.shape[1])
            graphs = graphs[:, edges_perm]

        traj_noise = torch.randn_like(trajs) * self.more_noise_std
        traj_noise[trajs == 0] = 0

        batch_data = {
            "trajs": trajs + traj_noise,
            "walks": walks,
            "graphs": graphs,
            "N_edges_per_graph": self.N_edges_per_graph[indices].to(DEVICE),
            "bbox": self.bboxes[indices].to(DEVICE)     # (min-x, min-y, max-x, max-y)
        }

        if self.need_heatmap:
            batch_data["heatmap"] = self.heatmaps[indices].to(DEVICE)
            batch_data["density_map"] = self.getDensityMaps(batch_data["trajs"]).to(DEVICE)

        if self.need_image:
            batch_data["image"] = self.images[indices].to(DEVICE)

        if self.need_nodes:
            batch_data["nodes"] = self.nodes[indices].to(DEVICE)
            batch_data["adj_mat"] = self.adj_mats[indices].to(DEVICE)
            batch_data["edges"] = self.edges[indices].to(DEVICE)
            batch_data["N_nodes"] = self.N_nodes[indices].to(DEVICE)
            batch_data["degrees"] = self.degrees[indices].to(DEVICE)

        if self.enable_aug:
            return self.augmentation(batch_data)
        return batch_data


    @staticmethod
    def getNodeHeatmaps(batch: dict):
        """
        Draw heatmap for the nodes
        :param batch: the batch of data
        :return: the heatmap of the nodes
        """
        B, _, H, W = batch["heatmap"].shape

        node_maps = torch.zeros((B, 1, H, W), dtype=torch.float32, device=DEVICE)

        for i in range(B):
            # Get the bounding box of the segment
            trajs = batch["trajs"][i]
            L_traj = batch["L_traj"][i]
            points = torch.cat([trajs[j, :L_traj[j]] for j in range(trajs.shape[0])], dim=0)
            min_point = torch.min(points, dim=0, keepdim=True).values
            max_point = torch.max(points, dim=0, keepdim=True).values
            point_range = max_point - min_point

            edges = batch["edges"][i]     # (N_edges, N_interp, 2)
            edges = edges[torch.all(edges.flatten(1) != 0, dim=1)]

            edges = (edges - min_point.view(1, 1, 2)) / point_range.view(1, 1, 2)

            edges[..., 0] = torch.clip(edges[..., 0] * W, 0, W-1)
            edges[..., 1] = torch.clip(edges[..., 1] * H, 0, H-1)

            seg_end_points = edges[:, [0, -1], :].flatten(0, 1)  # (2 * N_edges, 2)

            # fill seg end points pixels
            node_map = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)
            node_map[seg_end_points[:, 1].long(), seg_end_points[:, 0].long()] = 1

            node_maps[i, 0] = node_map

        return {"node_heatmap": node_maps}


    # @staticmethod
    # def sequencesToSegments(seqs: torch.Tensor, L_seg: int) -> torch.Tensor:
    #     # seqs: (B, N_seqs, L_seq, D_token)
    #     B, N_seqs, L_seq, D_token = seqs.shape
    # 
    #     result = torch.cat([
    #         seqs[:, :, :-1].view(B, N_seqs, -1, L_seg - 1, 2),  # (B, N_seqs, N_segs, L_seg-1, 2)
    #         seqs[:, :, L_seg - 1::L_seg - 1].unsqueeze(3)],  # (B, N_seqs, N_segs, 1, 2)
    #         dim=-2)
    # 
    #     # (B, N_seqs, N_segs, L_seg, 2)
    # 
    #     return result.flatten(-2, -1)  # (B, N_seqs, N_segs, L_seg * 2)


    def edgesToNodesAdj(self) -> None:
        B, N_edges, N_interp, _ = self.graphs.shape
        nodes_list = []
        adj_padded = []
        edges_padded = []
        nodes_counts = []
        inverse_id_list = []

        # 1. Get and count nodes for each graph
        end_points = self.graphs[:, :, [0, -1], :].flatten(1, 2)    # (B, N_end_points=2*N_edges, 2)
        for b in range(B):
            unique_nodes, inverse_indices = torch.unique(end_points[b], dim=0, return_inverse=True)
            nodes_list.append(unique_nodes)
            inverse_id_list.append(inverse_indices)
            nodes_counts.append(unique_nodes.shape[0])

        self.N_nodes = torch.tensor(nodes_counts, dtype=torch.long)
        max_node_count = int(self.N_nodes.max())
        self.max_N_nodes = max_node_count

        for b in tqdm(range(self.n_samples), desc="Building Nodes and Adjacency Matrices"):
            graph = self.graphs[b]  # Shape: (N_edges, N_interp, 2)

            pad_len = max_node_count - nodes_list[b].shape[0]
            nodes_list[b] = torch.nn.functional.pad(nodes_list[b], (0, 0, 0, pad_len), value=0.0)

            # Initialize adjacency matrix of size (nodes_pad_len, nodes_pad_len)
            adj_matrix = torch.zeros((max_node_count, max_node_count), dtype=torch.int32)
            edge_feature_matrix = torch.zeros((max_node_count, max_node_count, self.N_interp*2), dtype=torch.float32)

            # Fill adjacency matrix
            for j in range(N_edges):
                # Get the indices of the two points of the line segment in the unique nodes list
                p1_idx = inverse_id_list[b][2 * j]  # First point of the line segment
                p2_idx = inverse_id_list[b][2 * j + 1]  # Second point of the line segment
                adj_matrix[p1_idx, p2_idx] = 1
                adj_matrix[p2_idx, p1_idx] = 1  # Undirected edges

                # edge feature is the corresponding segment
                edge_feature_matrix[p1_idx, p2_idx] = graph[j].flatten()
                edge_feature_matrix[p2_idx, p1_idx] = graph[j].flip(0).flatten()

            adj_padded.append(adj_matrix)
            edges_padded.append(edge_feature_matrix)

        # Convert lists to tensors using torch.stack
        self.nodes = torch.stack(nodes_list)  # Shape: (B, nodes_pad_len, 2)
        self.adj_mats = torch.stack(adj_padded).to(torch.float32)  # Shape: (B, nodes_pad_len, nodes_pad_len)
        self.edges = torch.stack(edges_padded)  # Shape: (B, nodes_pad_len, nodes_pad_len, 16)
        self.degrees = torch.sum(self.adj_mats, dim=-1)     # (B, nodes_pad_len)


    # @staticmethod
    # def getJointsFromSegments(segments) -> Dict[str, torch.Tensor]:
    #     """
    #     Computes the adjacency (joint) matrix for a batch of line segments.
    #
    #     Args:
    #         segments (torch.Tensor): A tensor of shape (B, N, D), where D=5 (x1, y1, x2, y2, flag).
    #
    #     Returns:
    #         torch.Tensor: A joint matrix of shape (B, N, N) where each entry (i, j) is 1 if segments i and j are joint, 0 otherwise.
    #     """
    #     B, N, _ = segments.shape
    #
    #     p1 = segments[:, :, 0:2]    # (B, N, 2)
    #     p2 = segments[:, :, 2:4]    # (B, N, 2)
    #
    #     # p1p1_match[i, j] = 1 if p1[i] == p1[j]
    #     p1p1_match = torch.cdist(p1, p1) < 1e-5   # (B, N, N)
    #     p1p2_match = torch.cdist(p1, p2) < 1e-5   # (B, N, N)
    #     p2p1_match = torch.cdist(p2, p1) < 1e-5   # (B, N, N)
    #     p2p2_match = torch.cdist(p2, p2) < 1e-5   # (B, N, N)
    #
    #     # Combine the matches
    #     joint_matrix = p1p1_match | p1p2_match | p2p1_match | p2p2_match
    #
    #     return {"joints": joint_matrix.to(torch.float32)}


    def getDensityMaps(self, noisy_trajs: torch.Tensor):
        """
        Compute the density maps for the given trajectories
        :param noisy_trajs: the noisy trajectories of shape (B, N_trajs, L_traj, 2)
        :return: the density maps of shape (B, 1, H, W)
        """
        B, N_trajs, L_traj, _ = noisy_trajs.shape
        density_maps = torch.zeros((B, 1, self.img_H, self.img_W), dtype=torch.float32, device=DEVICE)

        for i in range(B):
            # Get the bounding box of the segment
            points = noisy_trajs[i].flatten(0, 1)  # (N_trajs * L_traj, 2)

            # During dataset generation, all points are shifted from bbox range to (-2, 2)
            # We just need to shift them back to (0, 1), then scale to (H, W)
            norm_points = (points + 2) / 4

            # Due to noise addition, some points may be out of range, we just eliminate them
            norm_valid_points = norm_points[(norm_points[:, 0] >= 0) & (norm_points[:, 0] <= 1) &
                                            (norm_points[:, 1] >= 0) & (norm_points[:, 1] <= 1)]

            x_ids = (norm_valid_points[:, 0] * (self.img_W - 1)).long()
            y_ids = (norm_valid_points[:, 1] * (self.img_H - 1)).long()

            flatten_ids = y_ids * self.img_W + x_ids
            flatten_map = torch.zeros(self.img_H * self.img_W, dtype=torch.float32, device=DEVICE)

            flatten_map.scatter_add_(0, flatten_ids, torch.ones_like(flatten_ids, dtype=torch.float32))
            density_maps[i, 0] = flatten_map.view(self.img_H, self.img_W) / flatten_map.max()

        return density_maps
