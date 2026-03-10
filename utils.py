from functools import partial
from pathlib import Path
from typing import Tuple, Callable

from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import Dataset

from dataset import ImagePuzzle
from model import Transformer, ImageSolver, TRM


def vanilla_step(model: nn.Module, optim: Optimizer, inputs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
	logits, = model(inputs)
	loss = None

	if labels is not None:
		loss = F.cross_entropy(logits.transpose(1, 2), labels)

		if optim is not None:
			loss.backward()
			optim.step()
			optim.zero_grad()

	return logits, loss


def vanilla(
		puzzle_size: int,
		model_dim: int,
		ff_dim: int,
		head_num: int,
		layer_num: int,
		**_,
) -> Tuple[nn.Module, Callable]:
	piece_num = puzzle_size * puzzle_size
	model = Transformer(model_dim, ff_dim, head_num, layer_num)
	model = ImageSolver(piece_num, model)
	step = vanilla_step
	return model, step


def trm_step(model: nn.Module, optim: Optimizer, inputs: Tensor, labels: Tensor, s: int) -> Tuple[Tensor, Tensor]:
	logits, loss, y, z = None, None, None, None

	for i in range(s):
		logits, y, z = model(inputs, y, z)
		if labels is not None:
			loss = F.cross_entropy(logits.transpose(1, 2), labels)

			if optim is not None:
				loss.backward()
				optim.step()
				optim.zero_grad()

	return logits, loss


def trm(
		puzzle_size: int,
		model_dim: int,
		ff_dim: int,
		head_num: int,
		layer_num: int,
		s: int,
		t: int,
		n: int,
		**_,
) -> Tuple[nn.Module, Callable]:
	piece_num = puzzle_size * puzzle_size
	model = TRM(model_dim, ff_dim, head_num, layer_num, piece_num, n, t)
	model = ImageSolver(piece_num, model)
	step = partial(trm_step, s=s)
	return model, step


MODELS = {
	"trm": trm,
	"vanilla": vanilla,
}


def get_model(name: str, **kwargs):
	model = MODELS[name](**kwargs)
	return model


def image(path: Path, puzzle_size: int, tile_size: int, **_) -> Tuple[Dataset, Dataset]:
	trainset = ImagePuzzle(path / "train", puzzle_size, tile_size)
	testset = ImagePuzzle(path / "test", puzzle_size, tile_size)
	return trainset, testset


load_dotenv()
DATASETS = os.getenv("DATASETS")
DATASETS = {
	"coco": (image, {"path": DATASETS / "COCO" / "2017"}),
}


def get_dataset(name: str, **kwargs):
	dataset, ds_kwargs = DATASETS[name]
	dataset = dataset(**(ds_kwargs | kwargs))
	return dataset
