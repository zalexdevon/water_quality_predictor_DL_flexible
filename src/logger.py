import logging
import os
from datetime import datetime


def configure_logger():
    logs_path = "artifacts/logs"
    os.makedirs(logs_path, exist_ok=True)

    date_format = "%d-%m-%Y-%H-%M-%S"
    log_file = f"{datetime.now().strftime(date_format)}.log"
    log_file_path = os.path.join(logs_path, log_file)

    logging.basicConfig(
        filename=log_file_path,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s\n %(message)s",
        level=logging.INFO,
        force=True,  # Needed to reconfigure logging if already set
    )
    return log_file_path
