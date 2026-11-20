
import logging
import sys


def main():
    # Configure timestamped logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    # Print active Python version
    print(f"Python Version: {sys.version}")

    # Initialize pipeline
    logger.info("Gesture Recognition Pipeline initialized.")


if __name__ == "__main__":
    main()
