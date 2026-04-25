"""GitHub API client with rate limiting, pagination, and checkpointing support."""

import random
import time
from typing import Any

import requests

from src.core.logger import get_logger

logger = get_logger(__name__)


class GitHubClient:
    """GitHub API client with robust rate limiting handling and checkpoint support."""

    BASE_URL = "https://api.github.com"

    # Rate limit management
    MIN_REQUEST_DELAY = 1.0  # Minimum seconds between requests
    MAX_REQUEST_DELAY = 10.0  # Maximum seconds between requests

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
        self._last_request_time = 0.0
        self._request_count = 0

    def _throttle(self) -> None:
        """
        Throttle requests to avoid hitting rate limits.
        Ensures minimum delay between consecutive requests.
        """
        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < self.MIN_REQUEST_DELAY:
            delay = self.MIN_REQUEST_DELAY + random.uniform(0, 0.5)  # Add jitter
            logger.debug(f"Throttling request by {delay:.2f}s")
            time.sleep(delay)

        self._last_request_time = time.time()
        self._request_count += 1

    def _request(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        """
        Make an HTTP request with retry logic for rate limits.
        """
        self._throttle()

        response = self.session.request(method, url, **kwargs)

        # Check rate limit headers
        remaining = int(response.headers.get("X-RateLimit-Remaining", 9999))
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        current_time = time.time()

        # Handle 403 Forbidden - includes both auth issues and rate limits
        if response.status_code == 403:
            retry_after = int(response.headers.get("Retry-After", 0))

            if remaining == 0 or "rate limit exceeded" in response.text.lower() or "abuse" in response.text.lower():
                # Use Retry-After header if available, otherwise calculate from reset_time
                if retry_after > 0:
                    wait_seconds = retry_after
                else:
                    wait_seconds = max(reset_time - current_time, 0) + 1

                logger.warning(f"Rate limit hit. Waiting {wait_seconds}s (Retry-After: {retry_after}, reset in {max(reset_time - current_time, 0):.0f}s)")
                time.sleep(min(wait_seconds, 300))

                # Make a fresh request after waiting - don't call raise_for_status()
                # as it would trigger retry and cause infinite loop
                logger.info("Making fresh request after rate limit wait")
                response = self.session.request(method, url, **kwargs)

                # Update remaining from new response
                remaining = int(response.headers.get("X-RateLimit-Remaining", 9999))
                logger.info(f"After wait - status: {response.status_code}, remaining: {remaining}")
            else:
                # Other 403 - likely auth issue
                logger.error(f"403 Forbidden (auth issue): {response.text[:200]}")
                response.raise_for_status()

        # Handle 429 Too Many Requests
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            wait_seconds = retry_after if retry_after > 0 else 60
            logger.warning(f"HTTP 429. Waiting {wait_seconds}s")
            time.sleep(wait_seconds)

            # Make fresh request after waiting
            response = self.session.request(method, url, **kwargs)
            remaining = int(response.headers.get("X-RateLimit-Remaining", 9999))
            logger.info(f"After 429 wait - status: {response.status_code}, remaining: {remaining}")

        # Handle 502/503 Server Errors
        if response.status_code in (502, 503):
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning(f"HTTP {response.status_code}. Waiting {retry_after}s")
            time.sleep(retry_after)
            response.raise_for_status()

        response.raise_for_status()
        return response.json()

    def search_repositories(
        self,
        query: str,
        per_page: int = 30,
        page: int = 1,
        sort: str = "stars",
        order: str = "desc",
    ) -> dict[str, Any]:
        """
        Search for repositories to act as a quality filter.
        """
        url = f"{self.BASE_URL}/search/repositories"
        params = {
            "q": query,
            "per_page": min(per_page, 100),
            "page": page,
            "sort": sort,
            "order": order,
        }

        logger.info(f"Searching Repos: {query} (page {page})")
        return self._request("GET", url, params=params)

    def search_code(
        self,
        query: str,
        per_page: int = 30,
        page: int = 1,
    ) -> dict[str, Any]:
        """
        Search for code using GitHub's search API.
        NOTE: /search/code does not support sorting.
        """
        url = f"{self.BASE_URL}/search/code"
        params = {
            "q": query,
            "per_page": min(per_page, 100),
            "page": page,
        }

        logger.info(f"Searching Code: {query} (page {page})")
        return self._request("GET", url, params=params)

    def get_file_content(self, repo: str, path: str, ref: str | None = None) -> dict[str, Any]:
        """Get the content of a file from a repository."""
        url = f"{self.BASE_URL}/repos/{repo}/contents/{path}"
        params = {"ref": ref} if ref else {}

        logger.debug(f"Fetching file: {repo}/{path}")
        return self._request("GET", url, params=params)

    def get_commits(self, repo: str, per_page: int = 30, sha: str | None = None) -> list[dict[str, Any]]:
        """Get commits for a repository."""
        url = f"{self.BASE_URL}/repos/{repo}/commits"
        params = {"per_page": per_page}
        if sha:
            params["sha"] = sha

        logger.debug(f"Fetching commits: {repo}")
        return self._request("GET", url, params=params)

    @staticmethod
    def score_kernel(item: dict[str, Any]) -> float:
        """
        Score a kernel search result for quality prioritization.
        Higher scores = more likely to be valuable production code.
        """
        score = 0.0

        repo = item.get("repository", {})
        stars = repo.get("stargazers_count", 0)
        score += stars * 0.1

        path = item.get("path", "").lower()
        filename = path.split("/")[-1]

        test_indicators = ["test", "example", "demo", "tutorial", "homework", "assignment", "practice"]
        for indicator in test_indicators:
            if indicator in path:
                score -= 20

        kernel_indicators = ["kernel", "/ops/", "/math/", "/cuda/", "/compute/", "/core/"]
        for indicator in kernel_indicators:
            if indicator in path:
                score += 15

        path_depth = path.count("/")
        if path_depth < 2:
            score -= 10
        elif path_depth >= 3:
            score += 5

        if filename.endswith(".cu") and not filename.endswith(".cuh"):
            score += 5

        return max(score, 0.0)

    def _sort_by_quality(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort search results by kernel quality score."""
        scored = [(self.score_kernel(item), item) for item in items]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]

    def search_cuda_files(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search for CUDA files by first finding highly-rated repositories,
        then searching for code within them. Results are sorted by quality.
        """
        all_items = []
        seen_signatures = set()

        repos_data = self.search_repositories(
            "language:CUDA stars:>50 fork:false",
            per_page=50
        )

        repos_items = repos_data.get("items", [])
        repos_items.sort(key=lambda x: x.get("stargazers_count", 0), reverse=True)
        top_repos = [(item["full_name"], item.get("stargazers_count", 0)) for item in repos_items]

        if not top_repos:
            logger.warning("No high-quality CUDA repositories found.")
            return []

        logger.info(f"Found {len(top_repos)} high-quality CUDA repos. Processing by star count.")

        for repo_name, repo_stars in top_repos:
            if len(all_items) >= max_results:
                break

            page = 1
            while len(all_items) < max_results:
                remaining = max_results - len(all_items)

                repo_query = f"{query} language:CUDA repo:{repo_name}"

                results = self.search_code(
                    query=repo_query,
                    per_page=min(30, remaining),
                    page=page,
                )

                items = results.get("items", [])
                if not items:
                    break

                for item in items:
                    item["repository"]["stargazers_count"] = repo_stars

                for item in items:
                    signature = (item.get("repository", {}).get("full_name"), item.get("path"))
                    if signature not in seen_signatures:
                        seen_signatures.add(signature)
                        all_items.append(item)

                logger.info(f"Fetched {len(items)} items from {repo_name} (total: {len(all_items)})")

                if len(items) < 30:
                    break

                page += 1
                delay = self.MIN_REQUEST_DELAY + random.uniform(0, 1)
                time.sleep(delay)

            delay = self.MIN_REQUEST_DELAY * 2 + random.uniform(0, 2)
            time.sleep(delay)

        all_items = self._sort_by_quality(all_items)
        return all_items[:max_results]

    def search_cuda_files_with_checkpoint(
        self,
        query: str,
        max_results: int = 100,
        checkpoint_data: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Search for CUDA files with checkpoint support for resume after crash/timeout.
        """
        repo_index = 0
        page = 1
        processed_count = 0
        seen_signatures = set()

        if checkpoint_data:
            if checkpoint_data.get("query") == query:
                repo_index = checkpoint_data.get("repo_index", 0)
                page = checkpoint_data.get("page", 1)
                processed_count = checkpoint_data.get("processed_count", 0)
                seen_signatures = set(checkpoint_data.get("seen_signatures", []))
                logger.info(f"Resuming: Repo index {repo_index}, page {page}, processed {processed_count}")
            else:
                logger.info("Query changed, starting fresh")

        all_items = []

        repos_data = self.search_repositories(
            "language:CUDA stars:>50 fork:false",
            per_page=50
        )

        repos_items = repos_data.get("items", [])
        repos_items.sort(key=lambda x: x.get("stargazers_count", 0), reverse=True)
        top_repos = [(item["full_name"], item.get("stargazers_count", 0)) for item in repos_items]

        if not top_repos:
            return [], {"query": query, "repo_index": 0, "page": 1, "processed_count": 0, "seen_signatures": []}

        while processed_count < max_results and repo_index < len(top_repos):
            repo_name, repo_stars = top_repos[repo_index]
            remaining = max_results - processed_count

            repo_query = f"{query} language:CUDA repo:{repo_name}"

            results = self.search_code(
                query=repo_query,
                per_page=min(30, remaining),
                page=page,
            )

            items = results.get("items", [])

            for item in items:
                item["repository"]["stargazers_count"] = repo_stars

            for item in items:
                signature = (item.get("repository", {}).get("full_name"), item.get("path"))
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    all_items.append(item)
                    processed_count += 1

            if items:
                logger.info(f"Fetched {len(items)} items from {repo_name} (total processed: {processed_count})")

            if len(items) == 30 and processed_count < max_results:
                page += 1
                delay = self.MIN_REQUEST_DELAY + random.uniform(0, 1)
                time.sleep(delay)
            else:
                repo_index += 1
                page = 1
                delay = self.MIN_REQUEST_DELAY * 2 + random.uniform(0, 2)
                time.sleep(delay)

        all_items = self._sort_by_quality(all_items)

        checkpoint = {
            "query": query,
            "repo_index": repo_index,
            "page": page,
            "processed_count": processed_count,
            "seen_signatures": list(seen_signatures),
        }

        return all_items[:max_results], checkpoint
