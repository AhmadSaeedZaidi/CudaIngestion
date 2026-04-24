"""GitHub API client with rate limiting, pagination, and checkpointing support."""

import time
from typing import Any, Tuple, List, Dict, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.logger import get_logger

logger = get_logger(__name__)


class GitHubClient:
    """GitHub API client with robust rate limiting handling and checkpoint support."""

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

    def search_repositories(
        self,
        query: str,
        per_page: int = 30,
        page: int = 1,
        sort: str = "stars",
        order: str = "desc",
    ) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
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

    def get_file_content(self, repo: str, path: str, ref: Optional[str] = None) -> Dict[str, Any]:
        """Get the content of a file from a repository."""
        url = f"{self.BASE_URL}/repos/{repo}/contents/{path}"
        params = {"ref": ref} if ref else {}

        logger.debug(f"Fetching file: {repo}/{path}")
        return self._request("GET", url, params=params)

    def get_commits(self, repo: str, per_page: int = 30, sha: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get commits for a repository."""
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
        Search for CUDA files by first finding highly-rated repositories,
        then searching for code within them.
        """
        all_items = []
        
        # Step 1: Fetch top-starred CUDA repositories
        repos_data = self.search_repositories("language:CUDA stars:>50", per_page=50)
        top_repos = [item["full_name"] for item in repos_data.get("items", [])]
        
        if not top_repos:
            logger.warning("No high-quality CUDA repositories found.")
            return []

        # Step 2: Search code within those specific repositories
        for repo_name in top_repos:
            if len(all_items) >= max_results:
                break
                
            page = 1
            while len(all_items) < max_results:
                remaining = max_results - len(all_items)
                
                # Scope query to the specific high-quality repo
                repo_query = f"{query} language:CUDA repo:{repo_name}"
                
                results = self.search_code(
                    query=repo_query,
                    per_page=min(30, remaining),
                    page=page,
                )

                items = results.get("items", [])
                if not items:
                    break # No more results in this repo

                all_items.extend(items)
                logger.info(f"Fetched {len(items)} items from {repo_name} (total: {len(all_items)})")

                if len(items) < 30:
                    break # Reached the last page for this repo

                page += 1
                time.sleep(2) # Respect secondary rate limits

            time.sleep(2) # Delay between repository switches

        return all_items[:max_results]

    def search_cuda_files_with_checkpoint(
        self,
        query: str,
        max_results: int = 100,
        checkpoint_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Search for CUDA files with checkpoint support for resume after crash/timeout.
        Implements the two-step repo-then-code architecture.
        """
        # Default state
        repo_index = 0
        page = 1
        processed_count = 0

        # Resume from checkpoint if valid
        if checkpoint_data:
            if checkpoint_data.get("query") == query:
                repo_index = checkpoint_data.get("repo_index", 0)
                page = checkpoint_data.get("page", 1)
                processed_count = checkpoint_data.get("processed_count", 0)
                logger.info(f"Resuming: Repo index {repo_index}, page {page}, processed {processed_count}")
            else:
                logger.info("Query changed, starting fresh")

        all_items = []
        
        # Step 1: Fetch stable list of top-starred repos to use as our pool
        repos_data = self.search_repositories("language:CUDA stars:>50", per_page=50)
        top_repos = [item["full_name"] for item in repos_data.get("items", [])]

        if not top_repos:
            return [], {"query": query, "repo_index": 0, "page": 1, "processed_count": 0}

        # Step 2: Iterate and search
        while processed_count < max_results and repo_index < len(top_repos):
            repo_name = top_repos[repo_index]
            remaining = max_results - processed_count

            repo_query = f"{query} language:CUDA repo:{repo_name}"

            results = self.search_code(
                query=repo_query,
                per_page=min(30, remaining),
                page=page,
            )

            items = results.get("items", [])
            
            if items:
                all_items.extend(items)
                processed_count += len(items)
                logger.info(f"Fetched {len(items)} items from {repo_name} (total processed: {processed_count})")

            # Pagination and repo-switching logic
            if len(items) == 30 and processed_count < max_results:
                page += 1
                time.sleep(2)
            else:
                # Exhausted this repo, move to the next one
                repo_index += 1
                page = 1 
                time.sleep(2)

        # Prepare checkpoint data representing the exact state to resume from
        checkpoint = {
            "query": query,
            "repo_index": repo_index,
            "page": page,
            "processed_count": processed_count,
        }

        return all_items[:max_results], checkpoint