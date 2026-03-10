import json
from collections.abc import Callable
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, Iterator, Iterable

import cv2
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImagePuzzle
from puzzle import fisher_yates
from utils import get_model, get_dataset


def run_epoch(
		model: nn.Module,
		optim: Optimizer,
		dataset: Iterable,
		step: Callable,
) -> Iterator[Tuple[Tensor, Tensor]]:
	train = optim is not None
	torch.set_grad_enabled(train)

	if train:
		model.train()
	else:
		model.eval()

	for inputs, labels in dataset:
		labels = fisher_yates(labels)
		labels = labels.cuda()
		inputs = inputs.cuda()

		logits, loss = step(model, optim, inputs, labels)
		logits = logits.detach()

		yield logits.cpu(), labels.cpu()


def run_setup(
		root: Path,
		epochs: int,
		model: nn.Module,
		step: Callable,
		optim: Optimizer,
		trainset: DataLoader,
		testset: DataLoader
) -> None:
	for epoch in range(2 * epochs):
		train = epoch % 2 == 0
		epoch = epoch // 2

		dataset = trainset if train else testset
		dataset = tqdm(dataset, desc=f"{epoch:02d}")
		ckpt = SimpleNamespace(epoch=epoch, train=train, logits=[], labels=[])

		for logits, labels in run_epoch(model, optim if train else None, dataset, step):
			ckpt.logits.append(logits)
			ckpt.labels.append(labels)

		ckpt.logits = torch.stack(ckpt.logits, dim=0)
		ckpt.labels = torch.stack(ckpt.labels, dim=0)

		if train:
			ckpt.model = model.state_dict()
			ckpt.optim = optim.state_dict()

		ckpt_path = root / ("train" if train else "test")
		ckpt_path.mkdir(parents=True, exist_ok=True)
		torch.save(ckpt, ckpt_path / f"{epoch:02d}.pt")


def run_name(init: str, params: dict) -> str:
	param_map = {
		"dataset": "",
		"puzzle_size": "P",
		"tile_size": "T",
		"model_dim": "D",
		"ff_dim": "F",
		"head_num": "H",
		"layer_num": "L",
		"s": "s",
		"t": "t",
		"n": "n",
	}
	parts = [init]
	params = params.copy()

	for long_key, short_key in param_map.items():
		val = params.pop(long_key, None)
		if val is None:
			continue
		parts.append(f"{short_key}{val}")

	if params:
		raise ValueError(f"Unknown params: {params}")

	return "-".join(parts)


def main() -> None:
	# Training params
	lr = 3e-4
	epochs = 10
	batch = 128
	workers = 4

	setups = [
		("trm", {
			"puzzle_size": [5, 6],
			"tile_size": [16],
			"model_dim": [128],
			"ff_dim": [512],
			"head_num": [8],
			"layer_num": [2],
			"s": [8],
			"t": [2],
			"n": [3],
			"dataset": ["coco"],
		}),
		# ("vanilla", {
		# 	"puzzle_size": [2, 3, 4, 5, 6],
		# 	"tile_size": [16],
		# 	"model_dim": [128],
		# 	"ff_dim": [512],
		# 	"head_num": [8],
		# 	"layer_num": [2, 4, 6, 8, 12],
		# 	"dataset": ["coco"],
		# })
	]

	for model_name, setup in setups:
		keys = list(setup.keys())
		vals = [setup[k] for k in keys]

		for params in product(*vals):
			params = dict(zip(keys, params))
			name = run_name(model_name, params)

			path = Path("run") / name
			if path.exists():
				continue

			path.mkdir(parents=True)
			print(name)

			with (path / "config.json").open("w") as f:
				json.dump({"model": model_name, "params": params}, f, indent=4)

			model, step = get_model(model_name, **params)
			model = model.cuda()
			optim = Adam(model.parameters(), lr=lr)

			dataset = params["dataset"]
			trainset, testset = get_dataset(dataset, **params)

			trainset = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True)
			testset = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=workers, drop_last=True)

			run_setup(path, epochs, model, step, optim, trainset, testset)


if __name__ == "__main__":
	main()
