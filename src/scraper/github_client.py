"""GitHub API client with rate limiting and pagination support."""

import time
from typing import Any, Dict, List, Optional
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from src.core.logger import get_logger

logger = get_logger(__name__)


class GitHubClient:
    """GitHub API client with robust rate limiting handling."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str):
        """
        Initialize GitHub client.

        Args:
            token: GitHub personal access token
        """
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    @retry(
        retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.Timeout)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic for rate limits.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            JSON response as dictionary
        """
        response = self.session.request(method, url, **kwargs)

        # Handle rate limiting (secondary rate limit)
        if response.status_code == 403:
            if "rate limit exceeded" in response.text.lower():
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                wait_seconds = max(reset_time - time.time(), 0) + 1
                logger.warning(f"Secondary rate limit hit. Waiting {wait_seconds:.0f}s")
                time.sleep(min(wait_seconds, 300))  # Cap at 5 minutes
                response.raise_for_status()

        # Handle specific error codes with backoff
        if response.status_code in (429, 502, 503):
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning(f"HTTP {response.status_code}. Retrying after {retry_after}s")
            time.sleep(retry_after)
            response.raise_for_status()

        response.raise_for_status()
        return response.json()

    def search_code(
        self,
        query: str,
        per_page: int = 30,
        page: int = 1,
        sort: Optional[str] = None,
        order: str = "desc",
    ) -> Dict[str, Any]:
        """
        Search for code using GitHub's search API.

        Args:
            query: Search query string
            per_page: Results per page (max 100)
            page: Page number
            sort: Sort field (indexed, stars, forks, etc.)
            order: Sort order (asc or desc)

        Returns:
            Search results with items and metadata
        """
        url = f"{self.BASE_URL}/search/code"
        params = {
            "q": query,
            "per_page": min(per_page, 100),
            "page": page,
            "order": order,
        }
        if sort:
            params["sort"] = sort

        logger.info(f"Searching GitHub: {query} (page {page})")
        return self._request("GET", url, params=params)

    def get_file_content(self, repo: str, path: str, ref: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the content of a file from a repository.

        Args:
            repo: Repository in 'owner/name' format
            path: Path to the file within the repository
            ref: Git reference (branch, commit, tag)

        Returns:
            File metadata including content (base64 encoded)
        """
        url = f"{self.BASE_URL}/repos/{repo}/contents/{path}"
        params = {"ref": ref} if ref else {}

        logger.debug(f"Fetching file: {repo}/{path}")
        return self._request("GET", url, params=params)

    def get_commits(self, repo: str, per_page: int = 30, sha: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get commits for a repository.

        Args:
            repo: Repository in 'owner/name' format
            per_page: Results per page
            sha: SHA or branch to start listing commits from

        Returns:
            List of commit objects
        """
        url = f"{self.BASE_URL}/repos/{repo}/commits"
        params = {"per_page": per_page}
        if sha:
            params["sha"] = sha

        logger.debug(f"Fetching commits: {repo}")
        return self._request("GET", url, params=params)

    def search_cuda_files(
        self,
        query: str,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search for CUDA files with automatic pagination.

        Args:
            query: Additional search query terms
            max_results: Maximum number of results to fetch

        Returns:
            List of search result items
        """
        all_items = []
        page = 1
        per_page = 30

        while len(all_items) < max_results:
            remaining = max_results - len(all_items)
            results = self.search_code(
                query=f"{query} language:CUDA",
                per_page=min(per_page, remaining),
                page=page,
            )

            items = results.get("items", [])
            if not items:
                break

            all_items.extend(items)
            logger.info(f"Fetched {len(items)} items (total: {len(all_items)})")

            # Check if we've reached the last page
            if len(items) < per_page or len(all_items) >= max_results:
                break

            page += 1
            # Be respectful of rate limits between pages
            time.sleep(2)

        return all_items[:max_results]
