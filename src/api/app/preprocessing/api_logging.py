import logging
import os

from dotenv import load_dotenv

load_dotenv()


logger: logging.Logger

if LOCAL_LOGGING := bool(int(os.getenv("LOCAL_LOGGING", 0))):
    # Use rich for local logging
    from rich.logging import RichHandler

    rich_handler = RichHandler(markup=True, rich_tracebacks=True)
    logger = logging.getLogger("insights")
    logger.addHandler(rich_handler)
    logger.handlers[0].setFormatter(logging.Formatter("%(message)s"))
else:
    logger = logging.getLogger("insights")
    handler = logging.StreamHandler()
    logger.addHandler(handler)

logger.setLevel(logging.DEBUG)
