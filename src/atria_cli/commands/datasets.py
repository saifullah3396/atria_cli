from __future__ import annotations

from typing import TYPE_CHECKING

import click
from atria_core.constants import _DEFAULT_ATRIA_DATASETS_CACHE_DIR
from atria_core.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from atria_datasets import FileStorageType

logger = get_logger(__name__)


def prepare_and_upload(
    name: str,
    config_name: str | None = None,
    branch: str = "main",
    is_public: bool = False,
    data_dir: str | None = None,
    provider: str | None = None,
    preprocess_transform: Callable | None = None,
    shard_storage_type: FileStorageType | None = None,
    access_token: str | None = None,
    num_processes: int = 0,
    dataset_load_mode: str = "in_memory",
    overwrite_existing_cached: bool = False,
    overwrite_existing_shards: bool = False,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    max_test_samples: int | None = None,
):
    """
    Uploads a dataset to the Atria Hub.
    """
    try:
        from atria_datasets import AtriaDataset

        logger.info(f"Loading dataset: {name}")
        dataset: AtriaDataset = AtriaDataset.load_from_registry(
            name=name,
            config_name=config_name,
            data_dir=data_dir,
            provider=provider,
            preprocess_transform=preprocess_transform,
            shard_storage_type=shard_storage_type,
            dataset_load_mode=dataset_load_mode,
            num_processes=num_processes,
            overwrite_existing_cached=overwrite_existing_cached,
            overwrite_existing_shards=overwrite_existing_shards,
            access_token=access_token,
            build_kwargs={
                "max_train_samples": max_train_samples,
                "max_validation_samples": max_validation_samples,
                "max_test_samples": max_test_samples,
            },
        )
        dataset.upload_to_hub(
            name=name.split("/")[-1] if name else name,
            branch=branch,
            is_public=is_public,
        )
    except Exception as e:
        logger.exception(e)


@click.command()
@click.option(
    "--name", required=True, type=str, help="Name of the dataset to download."
)
@click.option(
    "--config-name",
    required=False,
    type=str,
    default="default",
    help="Name of the config to download.",
)
@click.option(
    "--data-dir",
    required=False,
    type=str,
    default=None,
    help="Directory to download the dataset to.",
)
@click.option(
    "--overwrite-existing",
    is_flag=True,
    default=False,
    help="Whether to overwrite existing dataset.",
)
def download_dataset(
    name: str,
    config_name: str | None = "default",
    data_dir: str | None = None,
    overwrite_existing: bool = False,
):
    """
    Downloads a model from the Atria Hub.
    """
    from atria.data.datasets.atria_dataset import AtriaDataset

    try:
        if data_dir is None:
            data_dir = _DEFAULT_ATRIA_DATASETS_CACHE_DIR / name
            logger.info(f"Data directory not provided. Using default: {data_dir}")

        AtriaDataset.load_from_hub(
            name=name,
            config_name=config_name,
            data_dir=data_dir,
            overwrite_existing=overwrite_existing,
            streaming_mode=False,
        )
    except Exception as e:
        logger.exception("Failed to upload model:", exc_info=e)
