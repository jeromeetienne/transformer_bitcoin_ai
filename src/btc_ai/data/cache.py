import logging
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def url_to_cache_path(url: str, cache_dir: Path) -> Path:
        parsed = urlparse(url)
        return cache_dir / parsed.netloc / parsed.path.lstrip('/')


def download_to_cache(url: str, cache_dir: Path) -> Path:
        target = url_to_cache_path(url, cache_dir)
        if target.exists():
                logger.debug('cache hit: %s', target)
                return target

        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info('downloading %s', url)

        with requests.get(url, stream=True, timeout=30) as resp:
                if resp.status_code == 404:
                        raise FileNotFoundError(
                                f'404 not found: {url}\n'
                                f'For incomplete months, use period="daily" instead of "monthly".'
                        )
                resp.raise_for_status()

                tmp = tempfile.NamedTemporaryFile(
                        delete=False,
                        dir=target.parent,
                        prefix=f'.{target.name}.',
                        suffix='.tmp',
                )
                try:
                        with tmp:
                                for chunk in resp.iter_content(chunk_size=64 * 1024):
                                        tmp.write(chunk)
                        Path(tmp.name).rename(target)
                except BaseException:
                        Path(tmp.name).unlink(missing_ok=True)
                        raise

        return target
