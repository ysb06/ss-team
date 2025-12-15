import logging

from .exp import single_victim_experiment, double_victim_experiment

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)

if __name__ == "__main__":
    double_victim_experiment()