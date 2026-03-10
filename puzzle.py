import torch
from torch import Tensor


def fisher_yates(perm: Tensor) -> Tensor:
	b, n = perm.shape
	r = torch.arange(b)

	swaps = torch.zeros((b, n - 1), dtype=torch.long, device=perm.device)
	state = torch.arange(n, dtype=torch.long, device=perm.device)
	state = state[None, :].repeat(b, 1)

	for i in range(n - 1):
		x = perm[:, i, None]
		j = state.eq(x)

		j = j.int().argmax(dim=1)
		off = j - i

		swaps[:, i] = off
		state[:, i], state[r, j] = state[r, j].clone(), state[:, i].clone()

	return swaps


def inline(swaps: Tensor) -> Tensor:
	b, n = swaps.shape
	r = torch.arange(b)

	perm = torch.arange(n + 1, dtype=torch.long, device=swaps.device)
	perm = perm[None, :].repeat(b, 1)

	for i in range(n):
		s = swaps[:, i]
		s = i + s
		perm[:, i], perm[r, s] = perm[r, s].clone(), perm[:, i].clone()

	return perm
