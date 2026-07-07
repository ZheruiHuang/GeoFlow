import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphNodeEncoder(nn.Module):
    def __init__(self, in_feat_dim, hid_dim, out_dim, mlp_layers, gattn_layers):
        super().__init__()
        base_seq = []
        d_in = in_feat_dim
        for _ in range(mlp_layers - 1):
            base_seq += [nn.Linear(d_in, hid_dim), nn.ReLU()]
            d_in = hid_dim
        base_seq += [nn.Linear(d_in, hid_dim)]
        self.base_mlp = nn.Sequential(*base_seq)

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )

        self.struct_proj = nn.Linear(in_feat_dim + 2, hid_dim)
        self.gattn_layers = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(gattn_layers)])

        self.k_lin = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v_lin = nn.Linear(hid_dim, hid_dim, bias=False)

        self.mix_q = nn.Parameter(torch.zeros(3))
        self.tau_q = nn.Parameter(torch.tensor(1.0))

        self.fuse_layers = nn.Sequential(
            nn.Linear(hid_dim * 3, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def _build_struct_attn_and_value(self, cents, ap, h0):
        device, dtype = h0.device, h0.dtype
        A_geo = torch.as_tensor(ap["A_geo_norm"], dtype=dtype, device=device)
        A_hop = torch.as_tensor(ap["A_hop_norm"], dtype=dtype, device=device)
        A_ff  = torch.as_tensor(ap["A_ff_norm"],  dtype=dtype, device=device)

        K = self.k_lin(h0)  # (N,d)
        V = self.v_lin(h0)  # (N,d)

        wq = F.softmax(self.mix_q, dim=0)
        S_q = wq[0] * A_geo + wq[1] * A_hop + wq[2] * A_ff
        tau_q = F.softplus(self.tau_q) + 1e-6
        S_q = torch.softmax(S_q / tau_q, dim=-1)

        Pq = self.pos_mlp(cents)
        Q = S_q @ Pq

        d = Q.size(-1)
        logits = (Q @ K.transpose(0, 1)) / math.sqrt(d)
        A_attn = torch.softmax(logits, dim=-1)
        return A_attn, V

    def _gattn_forward(self, h_init, A_attn):
        h = h_init
        for layer in self.gattn_layers:
            h = A_attn @ h
            h = layer(h)
            h = F.relu_(h)
        return h

    def forward(self, feat_lst, centroid_lst, adj_pack_lst):
        """
        feat_lst: list[(N_i, F)]
        centroid_lst: list[(N_i, 2)]
        adj_pack_lst: list[dict], including "A_geo_norm", "A_hop_norm", "A_ff_norm"
        return: list[(N_i, d)]
        """
        outputs = []
        for feats, cents, ap in zip(feat_lst, centroid_lst, adj_pack_lst):
            base_emb = self.base_mlp(feats)                # (N,d)
            pos_emb  = self.pos_mlp(cents)                 # (N,d)
            feat_pos = torch.cat([feats, cents], dim=-1)
            h0 = self.struct_proj(feat_pos)                # (N,d)

            A_attn, V = self._build_struct_attn_and_value(cents, ap, h0)  # (N,N), (N,d)
            agg_emb = self._gattn_forward(V, A_attn)        # (N,d)

            fused = self.fuse_layers(torch.cat([base_emb, pos_emb, agg_emb], dim=-1))
            if self.residual:
                fused = fused + self.residual_proj(base_emb)
            outputs.append(fused)
        return outputs


class GatedGraphNodeEncoder(GraphNodeEncoder):
    def __init__(self, in_feat_dim, hid_dim, out_dim, mlp_layers, gattn_layers,
                 agg_gate_init=0.10, base_res_gate_init=0.50):
        super().__init__(in_feat_dim, hid_dim, out_dim, mlp_layers, gattn_layers)
        agg_gate_init = float(max(min(agg_gate_init, 1.0 - 1e-4), 1e-4))
        base_res_gate_init = float(max(min(base_res_gate_init, 1.0 - 1e-4), 1e-4))
        self.agg_gate_logit = nn.Parameter(torch.tensor(math.log(agg_gate_init / (1.0 - agg_gate_init))))
        self.base_res_gate_logit = nn.Parameter(torch.tensor(math.log(base_res_gate_init / (1.0 - base_res_gate_init))))
        self.base_res_proj = nn.Identity() if hid_dim == out_dim else nn.Linear(hid_dim, out_dim)

    def forward(self, feat_lst, centroid_lst, adj_pack_lst):
        outputs = []
        agg_gate = torch.sigmoid(self.agg_gate_logit)
        base_res_gate = torch.sigmoid(self.base_res_gate_logit)
        for feats, cents, ap in zip(feat_lst, centroid_lst, adj_pack_lst):
            base_emb = self.base_mlp(feats)
            pos_emb = self.pos_mlp(cents)
            feat_pos = torch.cat([feats, cents], dim=-1)
            struct_emb = self.struct_proj(feat_pos)

            struct_attn, value_emb = self._build_struct_attn_and_value(cents, ap, struct_emb)
            agg_emb = self._gattn_forward(value_emb, struct_attn) * agg_gate

            fused = self.fuse_layers(torch.cat([base_emb, pos_emb, agg_emb], dim=-1))
            outputs.append(fused + base_res_gate * self.base_res_proj(base_emb))
        return outputs

class SelfAttnBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.proj_dropout = nn.Dropout(p=dropout)

    def forward(self, x, attn_bias=None):
        # x: (N, d)
        N = x.size(0)
        h, d_h = self.n_heads, self.head_dim
        h_norm = self.norm1(x)
        q, k, v = self.qkv(h_norm).chunk(3, dim=-1)  # (N, d) x 3
        q = q.view(1, N, h, d_h).transpose(1, 2).contiguous()
        k = k.view(1, N, h, d_h).transpose(1, 2).contiguous()
        v = v.view(1, N, h, d_h).transpose(1, 2).contiguous()
        if attn_bias is not None:
            if attn_bias.dim() == 2:
                attn_mask = attn_bias.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
            else:
                attn_mask = attn_bias.unsqueeze(0)               # (1,h,N,N)
            attn_mask = attn_mask.to(dtype=q.dtype, device=q.device)
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=False
        )  # (1,h,N,d_h)
        out = out.transpose(1, 2).contiguous().view(N, self.dim)
        out = self.proj_dropout(self.proj(out))
        x = x + out
        x = x + self.ff(self.norm2(x))
        return x


class AxialPairAttnBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv_row = nn.Linear(dim, dim * 3)
        self.qkv_col = nn.Linear(dim, dim * 3)
        self.proj_row = nn.Linear(dim, dim)
        self.proj_col = nn.Linear(dim, dim)

        self.norm_in = nn.LayerNorm(dim)
        self.norm_mid = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def _attn_row(self, x):
        N, h, d_h = x.size(0), self.n_heads, self.head_dim
        qkv = self.qkv_row(x)           # (N,N,3d)
        q, k, v = qkv.chunk(3, dim=-1)  # (N,N,d)
        def split(t):
            return t.view(N, N, h, d_h).permute(0, 2, 1, 3).contiguous()  # (N,h,N,d_h)
        q, k, v = map(split, (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.drop.p if self.training else 0.0, is_causal=False
        )  # (N,h,N,d_h)
        out = out.permute(0, 2, 1, 3).contiguous().view(N, N, h * d_h)
        out = self.drop(self.proj_row(out))
        return out

    def _attn_col(self, x):
        N, h, d_h = x.size(0), self.n_heads, self.head_dim
        qkv = self.qkv_col(x)           # (N,N,3d)
        q, k, v = qkv.chunk(3, dim=-1)
        def split_swap(t):
            t = t.permute(1, 0, 2).contiguous()                  # (N_j,N_i,d)
            return t.view(N, N, h, d_h).permute(0, 2, 1, 3).contiguous()  # (N,h,N,d_h)
        q, k, v = map(split_swap, (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.drop.p if self.training else 0.0, is_causal=False
        )  # (N,h,N,d_h)
        out = out.permute(0, 2, 1, 3).contiguous().view(N, N, h * d_h)  # (N_j,N_i,d)
        out = out.permute(1, 0, 2).contiguous()                          # (N_i,N_j,d)
        out = self.drop(self.proj_col(out))
        return out

    def forward(self, x):
        # x: (N, N, d)
        y = self.norm_in(x)
        y = x + self._attn_row(y)
        y = self.norm_mid(y)
        y = y + self._attn_col(y)
        y = self.norm_out(y)
        y = y + self.ff(y)
        return y
