import torch
import torch.nn as nn
from layers.RevIN import RevIN
from scipy.signal import hilbert
from layers.Embed import DataEmbedding

class SFM(nn.Module):
    def __init__(self, enc_in, d_model, hidden_channels: int = 16,
                 kernel_size: int = 5, eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size should be odd, got {kernel_size}.")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}.")

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.eps = float(eps)
        self.enc_in = enc_in
        self.d_model = d_model
        self.tem = torch.nn.Parameter(torch.zeros(1,self.enc_in, self.d_model), requires_grad=True)

        # [log A_x, log A_t, cos φ_x, sin φ_x, cos φ_t, sin φ_t]
        self.hilbert_net = nn.Sequential(
            nn.Conv1d(
                in_channels=6,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # "same" padding
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            # nn.Conv1d(
            #     in_channels=hidden_channels,
            #     out_channels=1,
            #     kernel_size=kernel_size,
            #     padding=kernel_size // 2,
            # )
        )

    @staticmethod
    def _analytic_signal(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be (B, D, L), got {x.shape}.")
        x_np = x.detach().cpu().numpy()
        z_np = hilbert(x_np, axis=1)
        Z = torch.from_numpy(z_np).to(x.device)
        return Z

    @staticmethod
    def _check_input(x: torch.Tensor, t: torch.Tensor):
        if not torch.is_tensor(x) or not torch.is_tensor(t):
            raise TypeError("x and t must be torch.Tensor.")

        if x.shape != t.shape:
            raise ValueError(f"x and t must have the same shape, got {x.shape} and {t.shape}.")

        if x.ndim != 3:
            raise ValueError(f"x must be 3D (B, D, L), got {x.shape}.")
        if t.ndim != 3:
            raise ValueError(f"t must be 3D (B, D, L), got {t.shape}.")

        if not torch.is_floating_point(x) or not torch.is_floating_point(t):
            raise TypeError("x and t must be real-valued floating tensors.")

        if x.is_complex() or t.is_complex():
            raise TypeError("x and t must be real (not complex).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, L = x.shape
        device, dtype = x.device, x.dtype

        B = x.shape[0]
        t = self.tem.expand(B, -1, -1)
        self._check_input(x, t)

        x = x.contiguous()
        t = t.contiguous()

        # ====================SOR module====================
        # 1) z_x, z_t
        Zx = self._analytic_signal(x)  # (B, D, L), complex
        Zt = self._analytic_signal(t)  # (B, D, L), complex

        # 2) A_x, A_t
        Ax = torch.abs(Zx)  # (B, D, L), real
        At = torch.abs(Zt)  # (B, D, L), real

        Ax = Ax.clamp_min(self.eps)
        At = At.clamp_min(self.eps)

        # ====================SFGM module====================
        logAx = torch.log(Ax)  # (B, D, L)
        logAt = torch.log(At)  # (B, D, L)

        # 3)  cos φ, sin φ
        phix = torch.angle(Zx)  # (B, D, L), in [-π, π]
        phit = torch.angle(Zt)  # (B, D, L)

        cos_phix = torch.cos(phix)
        sin_phix = torch.sin(phix)
        cos_phit = torch.cos(phit)
        sin_phit = torch.sin(phit)

        # 4) [logAx, logAt, cosφx, sinφx, cosφt, sinφt]
        feat = torch.stack([logAx, logAt, cos_phix, sin_phix, cos_phit, sin_phit], dim=2, )
        feat = feat.view(B * D, 6, L)

        # 5) logits u
        u = self.hilbert_net(feat)  # (B*D, 1, L)
        u = u.view(B, D, L)         # (B, D, L)

        # ====================SRC module====================
        # 6) Sigmoid -> g ∈ (0,1)
        g = torch.sigmoid(u)  # (B, D, L)

        # 7) f = (1 - g) * x + g * t
        f = x + g * (t - x)

        f = f.to(dtype=dtype)
        return f

class FRM(nn.Module):
    def __init__(self,  d_model):
        super().__init__()
        self.d_model = d_model
        self.channelAggregator = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True,dropout=0.5)
        self.input_proj = nn.Sequential(nn.Linear(self.d_model, self.d_model),nn.GELU(),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn_out = self.channelAggregator(query=x, key=x, value=x)[0] + x
        x_proj_out = self.input_proj(x_attn_out)
        return x_proj_out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.revin_layer = RevIN(self.enc_in, affine=True)
        # self.hidden_inchanels = configs.hidden_channels
        # self.hidden_kernel = configs.hidden_kernel

        self.Enc_embedding = DataEmbedding(self.seq_len, self.d_model)
        self.SFM = SFM(enc_in=self.enc_in, d_model=self.d_model, hidden_channels=8, kernel_size=5)
        self.FRM = FRM(d_model=self.d_model)
        self.output_proj = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.pred_len),)


    def forward(self, x_enc, cycle_index):
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, mode='norm')

        # (1) Embedding
        x_enc_out = self.Enc_embedding(x_enc)

        # (2) SFM module
        x_sfm_out = self.SFM(x_enc_out)

        # (3) FRM
        x_frm_out = self.FRM(x_sfm_out)

        # (4) output
        output = self.output_proj(x_frm_out).permute(0, 2, 1)

        if self.use_revin:
            output = self.revin_layer(output, mode='denorm')
        return output

