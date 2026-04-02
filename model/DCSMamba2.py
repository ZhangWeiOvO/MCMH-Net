import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
mamba2-minimal
==============

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

> **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**
> Authors: Tri Dao, Albert Gu
> Paper: https://arxiv.org/abs/2405.21060
"""
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class Mamba2Simple(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            conv_init=None,
            expand=1,
            headdim=2,
            ngroups=1,
            A_init_range=(1, 16),
            dt_min=0.001,
            dt_max=0.1,
            dt_init_floor=1e-4,
            dt_limit=(0.0, float("inf")),
            learnable_init_states=False,
            activation="swish",
            bias=False,
            conv_bias=True,
            # Fused kernel and sharding options
            chunk_size=256,
            use_mem_eff_path=True,
            layer_idx=None,  # Absorb kwarg for general module
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        try:
            assert (self.d_model * self.expand / self.headdim) % 8 == 0
        except AssertionError:
            print(f"d_model: {self.d_model}, expand: {self.expand}, headdim: {self.headdim}")
            raise
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states = repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out


class DynamicGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = max(4, in_channels // 4)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),

            nn.Conv2d(mid_channels, 3, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.conv(x)  # [B,3,1,1]


# 修改点1：新增空间扫描函数 ----------------------------
class space_scan(nn.Module):
    def __init__(self):
        super().__init__()
        self.scan_cache = {}

    def generate_scan_index(self, H, W, device):
        key = (H, W)
        if key not in self.scan_cache:
            scan_idx = torch.empty(H * W, dtype=torch.long, device=device)
            idx = 0
            for row in range(H):
                cols = range(W) if row % 2 == 0 else reversed(range(W))
                for col in cols:
                    scan_idx[idx] = row * W + col
                    idx += 1
            self.scan_cache[key] = scan_idx
        return self.scan_cache[key]

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)
        scan_idx = self.generate_scan_index(H, W, x.device)
        return x_flat[:, :, scan_idx]


class DCSMamba2(nn.Module):
    def __init__(self, in_channels, d_model=16):
        super().__init__()

        self.gate_conv = DynamicGate(in_channels)
        self.space_scan = space_scan()
        # ================= 分支1：3x3核处理路径 =================
        self.branch3_conv = nn.Sequential(
            # 深度卷积（通道独立）
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            # 逐点卷积（通道融合）
            nn.Conv2d(in_channels, d_model, 1)
        )
        self.branch3_norm = nn.BatchNorm2d(d_model)
        self.branch3_mamba = Mamba2Simple(d_model)
        # 新增归一化层
        self.branch3_ln = nn.LayerNorm(d_model)

        # ================= 分支2：5x5核处理路径 =================
        self.branch5_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, d_model, 1)
        )
        self.branch5_norm = nn.BatchNorm2d(d_model)
        self.branch5_mamba = Mamba2Simple(d_model)
        # 新增归一化层
        self.branch5_ln = nn.LayerNorm(d_model)

        # ================= 分支3：7x7核处理路径 =================
        self.branch7_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, d_model, 1)
        )
        self.branch7_norm = nn.BatchNorm2d(d_model)
        self.branch7_mamba = Mamba2Simple(d_model)
        # 新增归一化层
        self.branch7_ln = nn.LayerNorm(d_model)

        # ================= 跨尺度注意力模块 =================
        self.attn_conv1 = nn.Conv2d(d_model * 3, d_model, 1)
        self.attn_conv2 = nn.Conv2d(d_model, 3, 3, padding=1)  # 生成3个注意力图
        self.sigmoid = nn.Sigmoid()

        # ================= 输出模块 =================
        self.output_conv = nn.Sequential(
            nn.Conv2d(d_model, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )
        self.res_weight = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(0.3)  # 在每一分支后使用Dropout防止过拟合

    def forward(self, x):
        identity = x

        # ============== 动态门控权重计算 ==============
        gate_weights = self.gate_conv(x)  # [B,3]
        w1 = gate_weights[:, 0].view(-1, 1, 1, 1)  # 分支1权重
        w2 = gate_weights[:, 1].view(-1, 1, 1, 1)  # 分支2权重
        w3 = gate_weights[:, 2].view(-1, 1, 1, 1)  # 分支3权重

        # ============== 分支1处理流程 ==============
        # 3x3分支
        b3_conv = self.branch3_conv(x)
        b3_norm = self.branch3_norm(b3_conv)
        B, C, H, W = b3_norm.shape
        b3_norm = self.space_scan(b3_norm)  # [B, d_model, L]
        b3_seq = b3_norm.view(B, C, -1).permute(0, 2, 1)
        b3_seq = self.branch3_ln(b3_seq)
        b3_seq = b3_seq.contiguous()
        b3_mamba = self.branch3_mamba(b3_seq)
        b3_out = b3_mamba.permute(0, 2, 1).view(B, C, H, W) * w1
        b3_out = self.dropout(b3_out)  # Dropout
        # ============== 分支2处理流程 ==============
        # 5x5分支
        b5_conv = self.branch5_conv(x)
        b5_norm = self.branch5_norm(b5_conv)
        b5_norm = self.space_scan(b5_norm)  # [B, d_model, L]
        b5_seq = b5_norm.view(B, C, -1).permute(0, 2, 1)
        b5_seq = self.branch5_ln(b5_seq)
        b5_seq = b5_seq.contiguous()
        b5_mamba = self.branch5_mamba(b5_seq)
        b5_out = b5_mamba.permute(0, 2, 1).view(B, C, H, W) * w2
        b5_out = self.dropout(b5_out)
        # ============== 分支3处理流程 ==============
        # 7x7分支
        b7_conv = self.branch7_conv(x)
        b7_norm = self.branch7_norm(b7_conv)
        b7_norm = self.space_scan(b7_norm)
        b7_seq = b7_norm.view(B, C, -1).permute(0, 2, 1)
        b7_seq = self.branch7_ln(b7_seq)
        b7_seq = b7_seq.contiguous()
        b7_mamba = self.branch7_mamba(b7_seq)
        b7_out = b7_mamba.permute(0, 2, 1).view(B, C, H, W) * w3
        b7_out = self.dropout(b7_out)
        # ============== 跨尺度注意力融合 ==============
        combined = torch.cat([b3_out, b5_out, b7_out], dim=1)
        attn_map = self.sigmoid(self.attn_conv2(self.attn_conv1(combined)))
        a1, a2, a3 = torch.chunk(attn_map, 3, dim=1)

        attended = (b3_out * a1) + (b5_out * a2) + (b7_out * a3)

        # ============== 残差连接 ==============
        output = self.output_conv(attended)
        return identity + self.res_weight * output

