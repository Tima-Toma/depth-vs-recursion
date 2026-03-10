import os
from enum import Enum
from pathlib import Path
from typing import Tuple, Iterator

import cv2
import numpy as np
from torch.utils.data import Dataset


def scan(path: str | Path, prefix: str = "") -> Iterator[str]:
	with os.scandir(path) as it:
		for item in it:
			if item.is_dir():
				yield from scan(item.path, os.path.join(prefix, item.name))
			else:
				yield os.path.join(prefix, item.name)


class ImagePuzzle(Dataset):
	class Resize(Enum):
		STRETCH = 0
		FIT = 1
		COVER = 2

	def __init__(
		self,
		root: str | Path,
		grid_size: int,
		tile_size: int,
		color: int = cv2.IMREAD_COLOR,
		inter: int = cv2.INTER_AREA,
		resize: Resize = Resize.STRETCH,
		seed: int = 42,
	) -> None:
		super().__init__()

		self.root = Path(root)
		self.grid_size = grid_size
		self.tile_size = tile_size
		self.color = color
		self.inter = inter
		self.resize = resize

		self.items = sorted(scan(root))
		self.random = np.random.default_rng(seed)

	def __len__(self) -> int:
		return len(self.items)

	def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
		rel_path = self.items[index]
		abs_path = self.root / rel_path

		image = cv2.imread(abs_path, self.color)
		image_size = self.grid_size * self.tile_size

		match self.resize:
			case ImagePuzzle.Resize.STRETCH:
				image = cv2.resize(image, (image_size, image_size), interpolation=self.inter)
			case ImagePuzzle.Resize.FIT:
				raise NotImplementedError()
			case ImagePuzzle.Resize.COVER:
				raise NotImplementedError()

		image = image.reshape(self.grid_size, self.tile_size, self.grid_size, self.tile_size, -1)
		image = image.transpose(0, 2, 4, 1, 3)
		image = image.reshape(-1, *image.shape[2:])
		image = image.astype(np.float32) / 255.0

		perm = self.random.permutation(image.shape[0])
		image = image[perm]

		return image, perm
