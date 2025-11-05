import pathlib
import asyncio
from typing import List, Tuple

import httpx
import aiofiles
from tqdm import tqdm


async def _download_file_async(
    url: str,
    dest_filepath: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    verbose: bool,
) -> Tuple[str, bool]:
    """
    The core download logic, running one task within the semaphore.
    This is fully asynchronous, using httpx for requests and aiofiles for disk I/O.
    """
    async with semaphore:
        if verbose:
            print(f"Starting download for {url}")
        for attempt in range(max_retries):
            try:
                pathlib.Path(dest_filepath).parent.mkdir(parents=True, exist_ok=True)
                async with client.stream(
                    "GET", url, timeout=30, follow_redirects=True
                ) as response:
                    response.raise_for_status()
                    async with aiofiles.open(dest_filepath, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            await f.write(chunk)
                    if verbose:
                        print(f"SUCCESS: {url} -> {dest_filepath}")
                    return dest_filepath, True  # Success
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                print(f"Attempt {attempt + 1}/{max_retries} FAILED for {url}: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s...
                    await asyncio.sleep(2**attempt)
                else:
                    if verbose:
                        print(f"PERMA-FAIL: {url} after {max_retries} attempts.")
                    return dest_filepath, False  # Final failure
    return dest_filepath, False  # Should be unreachable


async def download_files(
    files_to_download: List[Tuple[str, str]],
    max_concurrent: int = 10,
    max_rps: int = 5,
    max_retries: int = 3,
    verbose: bool = False,
    skip_existing: bool = True,
) -> List[Tuple[str, bool]]:
    """
    Downloads a batch of files concurrently with rate limiting and retries
    using httpx and aiofiles.

    Args:
        files_to_download: A list of (source_url, dest_filepath) tuples.
        max_concurrent: Max number of files to download at the same time.
        max_rps: Max number of new requests to start per second.
        max_retries: Max number of retries for each failed download.
        verbose: Show information about downloaded files. Default to False.
        skip_existing: Skip existing files. Default to True. Otherwise, redownload them.

    Returns:
        A list of (dest_filepath, success_boolean) tuples.
    """
    if verbose:
        print(
            f"Starting download for {len(files_to_download)} files. "
            f"Config: {max_concurrent} concurrent, {max_rps} RPS, {max_retries} retries."
        )
    semaphore = asyncio.Semaphore(max_concurrent)
    delay_between_requests = 1.0 / max_rps
    tasks = []
    async with httpx.AsyncClient() as client:
        for url, dest_filepath in files_to_download:
            if skip_existing and pathlib.Path(dest_filepath).exists():
                if verbose:
                    print(f"Skipping {dest_filepath}. File already exists.")
                continue
            task = asyncio.create_task(
                _download_file_async(url, dest_filepath, client, semaphore, max_retries, verbose)
            )
            tasks.append(task)
            await asyncio.sleep(delay_between_requests)
        results = await asyncio.gather(*tasks)
    if verbose:
        print("Batch download complete.")
        success_count = sum(1 for _, success in results if success)
        print(f"Successfully downloaded {success_count} / {len(results)} files.")
    return results
