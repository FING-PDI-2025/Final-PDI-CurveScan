import logging
from functools import cached_property
from pathlib import Path
from typing import Optional

import pandas as pd
from dataset_item import DatasetItem

log = logging.getLogger("dataset")


class Dataset:
    path: Path = Path(__file__).parent.parent / "datasets"

    def __init__(self):
        self.images = []
        self.images += [image for image in self.path.rglob("*.jpeg")]
        self.images += [image for image in self.path.rglob("*.jpg")]
        log.debug(f"Found {len(self.images)} images.")

    @cached_property
    def items(self) -> list[DatasetItem]:
        """
        Get the items in the dataset
        """
        return [DatasetItem(image, self) for image in self.images]

    @cached_property
    def df(self) -> pd.DataFrame:
        """
        Create a dataframe with the images and their properties
        """
        data = []
        for item in self.items:
            data.append(
                {
                    "name": item.name,
                    "path": item.path,
                    "type": item.type,
                    "has_flash": item.has_flash,
                    "has_light": item.has_light,
                }
            )
        return (
            pd.DataFrame(data)
            .sort_values(["name", "has_flash", "has_light"])
            .reset_index(drop=True)
        )

    @cached_property
    def pretty_df(self) -> pd.DataFrame:
        df = self.df.copy(deep=True)
        df["path"] = df["path"].apply(lambda x: x.name)
        return df

    def get(
        self, name: str, has_flash: bool = True, has_light: bool = True
    ) -> Optional["DatasetItem"]:
        """
        Get the image with the given name and properties
        """
        for item in self.items:
            if (
                item.name == name
                and item.has_flash == has_flash
                and item.has_light == has_light
            ):
                return item
        return None
