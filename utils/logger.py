import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(levelname)s %(name)s %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
