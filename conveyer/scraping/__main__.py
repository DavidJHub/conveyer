"""``python -m conveyer.scraping`` → run the scrape/classify pipeline."""

from .pipeline import _parse_args, run_scrape

if __name__ == "__main__":
    run_scrape(_parse_args())
