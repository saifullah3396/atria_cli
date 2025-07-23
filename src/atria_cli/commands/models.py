import os
from pathlib import Path

import click
from atria_core.logger import get_logger
from atriax_client.types import File

logger = get_logger(__name__)


@click.command()
@click.option(
    "--name",
    type=str,
    required=True,
    help=(
        "Name with which the model is registered in the Atria Hub. "
        "If not provided, the model name will be used as the registered name."
    ),
)
@click.option("--description", type=str, default=None, help="Description of the model.")
@click.option(
    "--is-public",
    is_flag=True,
    default=False,
    help="Whether the model is public or private.",
)
@click.option(
    "--ckpt-path",
    required=True,
    type=str,
    default=None,
    help="Path to the model checkpoint",
)
def upload_model(name: str, description: str, is_public: bool, ckpt_path: str = None):
    """
    Uploads a model to the Atria Hub.
    """
    try:
        from atriax_client.api.model import model_create
        from atriax_client.api.model.model_create import BodyModelCreate

        assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist"
        assert os.path.isfile(ckpt_path), f"Checkpoint path {ckpt_path} is not a file"
        with open(ckpt_path, "rb") as model_file:
            with client.authenticated_rest_client as client:
                response = model_create.sync_detailed(
                    client=client,
                    body=BodyModelCreate(
                        name=name,
                        model_file=File(
                            payload=model_file,
                            file_name="model.bin",
                            mime_type="application/octet-stream",
                        ),
                        description=description,
                        is_public=is_public,
                    ),
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to connect to atriax server: {response.status_code} - {response.content.decode()}"
                    )
                logger.info(
                    f"Model `{name}` uploaded successfully. Response: {response.parsed}"
                )
    except Exception as e:
        logger.exception("Failed to upload model:", exc_info=e)


@click.command()
@click.option(
    "--model-name", required=True, type=str, help="Name of the model to download."
)
@click.option(
    "--download-dir",
    required=False,
    type=str,
    default=None,
    help="Directory to download the model to.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Whether to overwrite the existing model.",
)
def download_model(model_name: str, download_dir: str, overwrite: bool = False):
    """
    Downloads a model from the Atria Hub.
    """
    try:
        from atria.hub.client import AtriaXClient
        from atria.hub.config import settings
        from atria_core.constants import _DEFAULT_ATRIA_MODELS_CACHE_DIR

        download_dir = download_dir or str(_DEFAULT_ATRIA_MODELS_CACHE_DIR)
        output_path = Path(download_dir) / model_name / "model.bin"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            logger.info(
                f"Model already exists at `{output_path}`. Skipping download. Set `--overwrite` to force replace."
            )
            return

        client = AtriaXClient(
            base_url=settings.ATRIAX_URL,
            anon_api_key=settings.ATRIAX_ANON_KEY,
            initialize_auth=False,
        )

        logger.info(f"Downloading model `{model_name}` to `{output_path}`")
        client.model.load(model_name=model_name, destination_path=output_path)

    except Exception as e:
        logger.exception("Failed to upload model:", exc_info=e)
