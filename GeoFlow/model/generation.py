import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import SelfAttnBlock, AxialPairAttnBlock, GatedGraphNodeEncoder


REL_FEATURE_DIM = 5


def timestep_embedding(timesteps, dim, max_period=10000):
    if not torch.is_tensor(timesteps):
        timesteps = torch.as_tensor(timesteps)
    timesteps = timesteps.to(dtype=torch.float32)
    device = timesteps.device
    timesteps = timesteps.view(-1)

    assert dim % 2 == 0
    half = dim // 2
    if half == 0:
        return torch.zeros((timesteps.shape[0], 0), device=device)

    freq_exponent = torch.arange(half, device=device, dtype=torch.float32) / half
    freqs = torch.exp(-math.log(max_period) * freq_exponent)

    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return emb


class GlobalVelocityNet(nn.Module):
    def __init__(self, feat_dim, time_dim, hid_dim, n_heads, attn_layers,
                 axial_attn_layers, dropout):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, hid_dim)
        self.node_attn_layers = nn.ModuleList(
            [SelfAttnBlock(hid_dim, n_heads, dropout) for _ in range(attn_layers)]
        )
        self.rel2attn = nn.Sequential(nn.Linear(3, n_heads), nn.Tanh())
        self.origin_proj = nn.Linear(hid_dim, hid_dim)
        self.dest_proj = nn.Linear(hid_dim, hid_dim)
        self.rel_mlp = nn.Sequential(nn.Linear(REL_FEATURE_DIM, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
        self.xt_mlp = nn.Sequential(nn.Linear(1, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))

        self.lin_o   = nn.Linear(hid_dim, hid_dim, bias=False)
        self.lin_d   = nn.Linear(hid_dim, hid_dim, bias=False)
        self.lin_od  = nn.Linear(hid_dim, hid_dim, bias=False)
        self.lin_rel = nn.Linear(hid_dim, hid_dim, bias=False)
        self.lin_xt  = nn.Linear(hid_dim, hid_dim, bias=False)
        self.lin_t   = nn.Linear(hid_dim, hid_dim, bias=False)

        self.pair_axial_layers = nn.ModuleList(
            [AxialPairAttnBlock(hid_dim, n_heads, dropout) for _ in range(axial_attn_layers)]
        )
        self.pair2node = nn.Sequential(
            nn.LayerNorm(hid_dim * 2),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(hid_dim * 6, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, xt_list, feat_emb_lst, centroid_lst, adj_pack_lst, t_emb):
        outputs = []
        for idx, (xt, h0, cents, ap) in enumerate(zip(xt_list, feat_emb_lst, centroid_lst, adj_pack_lst)):
            t_graph = self.time_proj(t_emb[idx:idx+1])
            h0 = h0 + t_graph

            delta = cents.unsqueeze(1) - cents.unsqueeze(0)
            dist = torch.norm(delta, dim=-1, keepdim=True)
            attn_bias = self.rel2attn(torch.cat([delta, dist], -1)).permute(2, 0, 1).contiguous()

            aff_ff  = ap["aff_ff"].unsqueeze(-1)
            aff_hop = ap["aff_hop"].unsqueeze(-1)
            rel_emb = self.rel_mlp(torch.cat([delta, dist, aff_ff, aff_hop], dim=-1))
            xt_emb = self.xt_mlp(xt.unsqueeze(-1))
            t_pair = t_graph.reshape(1, 1, -1).expand_as(rel_emb)

            o0 = self.origin_proj(h0)
            d0 = self.dest_proj(h0)
            o0e = o0.unsqueeze(1)
            d0e = d0.unsqueeze(0)
            od0 = o0e * d0e
            pair_emb = (
                self.lin_o(o0e.expand(-1, h0.size(0), -1)) +
                self.lin_d(d0e.expand(h0.size(0), -1, -1)) +
                self.lin_od(od0) +
                self.lin_rel(rel_emb) +
                self.lin_xt(xt_emb) +
                self.lin_t(t_pair)
            )

            for layer in self.pair_axial_layers:
                pair_emb = layer(pair_emb)

            h_o = pair_emb.mean(dim=1)
            h_d = pair_emb.mean(dim=0)
            h_aggr = self.pair2node(torch.cat([h_o, h_d], dim=-1))

            h = h0 + h_aggr
            for layer in self.node_attn_layers:
                h = layer(h, attn_bias=attn_bias)

            o = self.origin_proj(h)
            d = self.dest_proj(h)
            oe = o.unsqueeze(1)
            de = d.unsqueeze(0)
            od = oe * de
            pair_feat = torch.cat([
                oe.expand(-1, h.size(0), -1),
                de.expand(h.size(0), -1, -1),
                od,
                rel_emb,
                xt_emb,
                t_pair,
            ], dim=-1)
            v = self.pair_mlp(pair_feat).squeeze(-1)
            del pair_feat
            outputs.append(v)
        return outputs


class FMODGModel(nn.Module):
    def __init__(self, feat_dim, hid_dim, time_dim, encode_mlp_layers,
                 encode_gattn_layers, attn_heads, attn_layers, axial_attn_layers,
                 dropout):
        super().__init__()
        self.time_dim = time_dim
        self.encoder = GatedGraphNodeEncoder(
            feat_dim, hid_dim, hid_dim, encode_mlp_layers, encode_gattn_layers
        )
        self.proj_time = nn.Linear(time_dim, time_dim)
        self.vel_net = GlobalVelocityNet(hid_dim, time_dim, hid_dim,
                                          n_heads=attn_heads,
                                          attn_layers=attn_layers,
                                          axial_attn_layers=axial_attn_layers,
                                          dropout=dropout)
    
    def _time_embed(self, t):
        """
        ts: (B,) or (B, 1)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        emb = timestep_embedding(t, self.time_dim)  # (B, d_t)
        emb = self.proj_time(emb)  # (B, d_t)
        return emb
    
    def forward(self, xt_list, feat_lst, centroid_lst, adj_pack_lst, t_lst):  # for flow-matching model
        """
        feat_lst: list of B features, each of shape (N_i, F)
        xt_list: list[(N_i, N_i)]
        t_lst: list of time steps
        return: list of velocity fields, each of shape (N_i, N_i)
        """
        ts = torch.cat(t_lst)  # (B,)
        t_emb = self._time_embed(ts)  # (B, d_t)
        feat_emb_lst = self.encoder(feat_lst, centroid_lst, adj_pack_lst)  # list of (N_i, d)
        v = self.vel_net(xt_list, feat_emb_lst, centroid_lst, adj_pack_lst, t_emb)
        return v


class FlowMatchingTrainer:
    def __init__(self, model, lr, device, grad_accum_steps):
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self._accum_counter = 0

    def _sample_interpolation(self, od_matrix_lst):
        """
        Sample interpolation points from the OD matrix.
        od_matrix_list: list of (N_i, N_i) matrices
        return: x_t, v_target, t
        """
        xt_lst, v_tgt_lst, t_lst = [], [], []
        for mat in od_matrix_lst:
            x1 = mat
            x0 = torch.randn_like(x1)
            t = torch.rand(1, device=mat.device)
            x_t = (1-t)*x0 + t*x1
            v_target = x1 - x0
            xt_lst.append(x_t)
            v_tgt_lst.append(v_target)
            t_lst.append(t)
        return xt_lst, v_tgt_lst, t_lst

    def train_step(self, feat_lst, centroid_lst, adj_pack_lst, od_matrix_lst, mask_lst=None):
        """
        Train the model for one step.
        feat_lst: list of B features, each of shape (N_i, F)
        od_matrix_lst: list of (N_i, N_i) matrices
        return: loss value
        """
        self.model.train()
        if self._accum_counter % self.grad_accum_steps == 0:
            self.optimizer.zero_grad()

        x_t, v_target, t = self._sample_interpolation(od_matrix_lst)
        v_pred = self.model(x_t, feat_lst, centroid_lst, adj_pack_lst, t)
        loss = 0.
        if mask_lst is None:
            mask_iter = [None] * len(v_pred)
        else:
            mask_iter = mask_lst
        for v, v_tgt, x1, mask in zip(v_pred, v_target, od_matrix_lst, mask_iter):
            if mask is not None:
                pos_mask = (x1 > 0) & (mask > 0)
                if pos_mask.any():
                    loss += F.mse_loss(v[pos_mask], v_tgt[pos_mask])
                else:
                    loss += F.mse_loss(v, v_tgt)
            else:
                loss += F.mse_loss(v, v_tgt)
        loss = loss / len(v_pred)

        loss_value = loss.detach().item()
        (loss / self.grad_accum_steps).backward()
        self._accum_counter += 1

        if self._accum_counter % self.grad_accum_steps == 0:
            self.optimizer.step()

        return loss_value

    @torch.no_grad()
    def sample(self, feat_lst, centroid_lst, adj_pack_lst, n_step):
        """
        Euler sampling.
        feat_lst: list of B features, each of shape (N_i, F)
        n_step: number of sampling steps
        return: list of sampled OD matrices, each of shape (N_i, N_i) and list of time steps
        """
        self.model.eval()
        xt_lst = [torch.randn([feat.size(0), feat.size(0)], device=feat.device) for feat in feat_lst]
        ts = [torch.zeros(1, device=feat.device) for feat in feat_lst]
        dt = 1. / n_step
        for _ in range(n_step):
            v_lst = self.model(xt_lst, feat_lst, centroid_lst, adj_pack_lst, ts)
            xt_lst = [xt + v * dt for xt, v in zip(xt_lst, v_lst)]
            ts = [t + dt for t in ts]
        return xt_lst, ts
