import fire
from atria_core.logger import get_logger

from atria_cli.commands.datasets import download, prepare_and_upload
from atria_cli.commands.sign_in import sign_in
from atria_cli.commands.sign_out import sign_out
from atria_cli.commands.sign_up import sign_up

logger = get_logger(__name__)


if __name__ == "__main__":
    fire.Fire(
        {
            "sign_in": sign_in,
            "sign_out": sign_out,
            "sign_up": sign_up,
            "datasets": {
                "prepare_and_upload": prepare_and_upload,
                "download": download,
            },
        }
    )
