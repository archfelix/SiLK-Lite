import torch
import torch.nn.functional as F


class DynamicCosineSim():
    def __init__(self, desc0: torch.Tensor, desc1: torch.Tensor, block_size=100, tau=1):
        B, D, H, W = desc0.shape
        assert B == 1
        self.device = desc0.device
        self.block_size = block_size

        self.desc0 = desc0.view(D, H*W).permute(1, 0)
        self.desc1 = desc1.view(D, H*W).permute(1, 0)
        self.desc0 = self.desc0 / torch.norm(self.desc0, p=2, dim=1, keepdim=True)
        self.desc1 = self.desc1 / torch.norm(self.desc1, p=2, dim=1, keepdim=True)

        self.D = D
        self.H = H
        self.W = W
        self.M = H*W
        self.tau = tau

        self.precompute_sum_Sik_sum_Skj()

    def precompute_sum_Sik_sum_Skj(self):
        # cosine_similarity is too slow and takes large memory, so use MatMul
        # desc0 = self.desc0.reshape(self.M, 1, self.D)
        # desc1 = self.desc1.reshape(1, self.M, self.D)
        # self.sim_mat = F.cosine_similarity(desc0, desc1, dim=2)
        self.sim_mat = torch.matmul(self.desc0, self.desc1.T)

        self.max_sik = torch.max(self.sim_mat, dim=0)[0]
        self.max_skj = torch.max(self.sim_mat, dim=1)[0]

        # Old:
        # self.sum_sik = torch.sum(torch.exp(self.sim_mat / self.tau), dim=0)
        # self.sum_skj = torch.sum(torch.exp(self.sim_mat / self.tau), dim=1)

        # New:
        self.sum_sik = torch.logsumexp(self.sim_mat / self.tau, dim=0)
        self.sum_skj = torch.logsumexp(self.sim_mat / self.tau, dim=1)

    def get_sim_ij(self, pos: torch.Tensor):
        # pos.shape=[N, 2]
        return torch.sum(self.desc0[pos[:, 0], :] * self.desc1[pos[:, 1], :], dim=1)

    # Old:
    # def get_Pi_to_j(self, pos: torch.Tensor):
    #     # pos.shape=[N, 2]
    #     # this is just a softmax
    #     return torch.exp(self.get_sim_ij(pos) / self.tau) / self.sum_sik[pos[:, 0]]

    # New:
    def get_log_Pi_to_j(self, pos: torch.Tensor):
        # pos.shape=[N, 2]
        return (self.get_sim_ij(pos) / self.tau) - self.sum_sik[pos[:, 0]]

    # Old:
    # def get_Pi_from_j(self, pos: torch.Tensor):
    #     # pos.shape=[N, 2]
    #     # this is just a softmax
    #     return torch.exp(self.get_sim_ij(pos) / self.tau) / self.sum_skj[pos[:, 1]]

    # New:
    def get_log_Pi_from_j(self, pos: torch.Tensor):
        # pos.shape=[N, 2]
        return (self.get_sim_ij(pos) / self.tau) - self.sum_skj[pos[:, 1]]

    # Old:
    # def get_Pij(self, pos: torch.Tensor):
    #     # self.get_Pij(torch.tensor([[i, j],
    #     #                            [0, 1],
    #     #                            [0, 2]]))
    #     return torch.log(self.get_Pi_to_j(pos)) + torch.log(self.get_Pi_from_j(pos))

    # New:
    def get_Pij(self, pos: torch.Tensor):
        # self.get_Pij(torch.tensor([[i, j],
        #                            [0, 1],
        #                            [0, 2]]))
        return self.get_log_Pi_to_j(pos) + self.get_log_Pi_from_j(pos)
