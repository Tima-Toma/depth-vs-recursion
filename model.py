from typing import Tuple, List, Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
	def __init__(self, model_dim: int, ff_dim: int) -> None:
		super().__init__()
		self.gate_proj = nn.Linear(model_dim, 2 * ff_dim, bias=False)
		self.down_proj = nn.Linear(ff_dim, model_dim, bias=False)

	def forward(self, x: Tensor) -> Tensor:
		gate = self.gate_proj(x)
		gate, up = gate.chunk(2, dim=-1)
		down = self.down_proj(F.silu(gate) * up)
		return down


class EncoderBlock(nn.Module):
	def __init__(self, model_dim: int, head_num: int, ff_dim: int) -> None:
		super().__init__()
		self.attn = nn.MultiheadAttention(model_dim, head_num, dropout=0.1, batch_first=True, bias=False)
		self.ff = SwiGLU(model_dim, ff_dim)

	def forward(self, x: Tensor) -> Tensor:
		attn, _ = self.attn(x, x, x)
		x = F.rms_norm(x + attn, [x.shape[-1]], eps=1e-6)
		x = F.rms_norm(x + self.ff(x), [x.shape[-1]], eps=1e-6)
		return x


class FisherYatesHead(nn.Module):
	def __init__(self, model_dim: int, piece_num: int) -> None:
		super().__init__()
		# Linear layers with decreasing output sizes:
		# Step 0: output size = piece_num
		# Step 1: output size = piece_num - 1
		# ...
		self.ff = nn.ModuleList([
			nn.Linear(model_dim, piece_num - i)
			for i in range(piece_num - 1)
		])

	def forward(self, x: Tensor) -> Tensor:
		B, S, D = x.shape
		logits = torch.full((B, S - 1, S), -torch.inf, device=x.device)

		for i in range(S - 1):
			logits[:, i, :S - i] = self.ff[i](x[:, i, :])

		return logits


class Transformer(nn.Module):
	def __init__(self, model_dim: int, ff_dim: int, head_num: int, layer_num: int) -> None:
		super().__init__()
		self.model_dim = model_dim
		self.net = nn.Sequential(*[EncoderBlock(model_dim, head_num, ff_dim) for _ in range(layer_num)])

	def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
		return self.net(x),


# Taken from HRM (https://github.com/raincchio/HRM-mini/blob/main/2_hrm_mini_train.ipynb),
def trunc_normal_init(x: torch.Tensor, std: float):
	return nn.init.trunc_normal_(x, std=std).mul_(1.1368472343385565)


class TRM(nn.Module):
	def __init__(
			self,
			model_dim: int,
			ff_dim: int,
			head_num: int,
			layer_num: int,
			piece_num: int,
			n: int,
			t: int
	) -> None:
		super().__init__()
		self.model_dim = model_dim
		self.n = n
		self.t = t
		self.net = nn.Sequential(*[EncoderBlock(model_dim, head_num, ff_dim) for _ in range(layer_num)])
		self.y_init = nn.Buffer(trunc_normal_init(torch.empty(1, piece_num, model_dim), std=1))
		self.z_init = nn.Buffer(trunc_normal_init(torch.empty(1, piece_num, model_dim), std=1))

	def _latent(self, x: Tensor, y: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
		for i in range(self.n):
			z = self.net(x + y + z)
		y = self.net(y + z)
		return y, z

	def _deep(self, x: Tensor, y: Tensor, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
		with torch.no_grad():
			for j in range(self.t - 1):
				y, z = self._latent(x, y, z)
		y, z = self._latent(x, y, z)
		return y, y.detach(), z.detach()

	def forward(self, x: Tensor, y: Optional[Tensor], z: Optional[Tensor]) -> Tuple[Tensor, ...]:
		B, S, _ = x.shape
		y = self.y_init.expand(B, S, -1) if y is None else y
		z = self.z_init.expand(B, S, -1) if z is None else z
		x, y, z = self._deep(x, y, z)
		return x, y, z


class ImageSolver(nn.Module):
	def __init__(self, piece_num: int, core: nn.Module) -> None:
		super().__init__()
		self.backbone = nn.Sequential(
			nn.LazyConv2d(32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 5, 1, 2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Flatten(),
			nn.LazyLinear(core.model_dim),
		)
		self.core = core
		self.head = FisherYatesHead(core.model_dim, piece_num)
		self.piece_pos = nn.Parameter(torch.empty(piece_num, core.model_dim))
		nn.init.uniform_(self.piece_pos)

	def forward(self, x: Tensor, *args) -> Tuple[Tensor, ...]:
		B, S, C, H, W = x.shape

		x = x.view(B * S, C, H, W)
		x = self.backbone(x)
		x = x.view(B, S, -1)
		x = x + self.piece_pos[None, :, :]

		x, *state = self.core(x, *args)
		x = self.head(x)

		return x, *state
