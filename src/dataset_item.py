from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import cv2 as cv
import numpy as np


@dataclass
class DatasetItem:
    path: Path
    _base_dataset: "Dataset"

    @cached_property
    def name(self) -> str:
        """The name of the image without [flash] or [light] terms"""
        stem = self.path.stem
        parts = stem.split("_")
        for column in ["flash", "light"]:
            if "no" + column in parts:
                parts.remove("no" + column)
            if column in parts:
                parts.remove(column)
        return "_".join(parts)

    @cached_property
    def has_flash(self) -> bool:
        if not "flash" in self.path.stem.lower():
            raise ValueError(f"Image {self.path} does not have flash in its name")
        return "noflash" not in self.path.stem.lower()

    @cached_property
    def has_light(self) -> bool:
        if not "light" in self.path.stem.lower():
            raise ValueError(f"Image {self.path} does not have light in its name")
        return "nolight" not in self.path.stem.lower()

    @cached_property
    def type(self) -> str:
        """
        Get the type of the image based on its name
        """
        return self.path.parent.name

    @property
    def with_flash(self) -> Self:
        """
        Get the image with flash
        """
        return self._base_dataset.get(
            self.name, has_flash=True, has_light=self.has_light
        )

    @property
    def with_light(self) -> Self:
        """
        Get the image with light
        """
        return self._base_dataset.get(
            self.name, has_flash=self.has_flash, has_light=True
        )

    @property
    def without_flash(self) -> Self:
        """
        Get the image without flash
        """
        return self._base_dataset.get(
            self.name, has_flash=False, has_light=self.has_light
        )

    @property
    def without_light(self) -> Self:
        """
        Get the image without light
        """
        return self._base_dataset.get(
            self.name, has_flash=self.has_flash, has_light=False
        )

    @property
    def other_variants(self) -> list[Self]:
        """
        Get the other variants of the image
        """
        return [
            item
            for item in self._base_dataset.items
            if item.name == self.name and item.path != self.path
        ]

    @cached_property
    def data(self) -> np.ndarray:
        """
        Get the image data
        """
        result = cv.imread(str(self.path.absolute()), cv.IMREAD_UNCHANGED)
        if result is None:
            raise ValueError(f"Image {self.path.absolute()} not found")
        return result
