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

    # Rate limit management - reduced delays for better performance
    MIN_REQUEST_DELAY = 1.0  # 1 second between requests
    REQUEST_DELAY_JITTER = 0.5  # Add up to 0.5s random jitter

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

        # Track rate limit state to prevent repeated hits
        self._rate_limit_reset_at = 0.0  # Unix timestamp when rate limit resets
        self._last_remaining = 9999  # Last known remaining requests

    def _wait_if_rate_limited(self) -> None:
        """
        Proactively wait if we're in a rate limit window.
        Called BEFORE making any request.
        """
        current_time = time.time()

        # If we're within the rate limit window, wait until it resets
        if self._rate_limit_reset_at > current_time:
            wait_time = self._rate_limit_reset_at - current_time + 1
            logger.info(f"Proactive wait: in rate limit window, waiting {wait_time:.0f}s until reset")
            time.sleep(min(wait_time, 300))
            current_time = time.time()

        # Only add extra delay when we're critically low on requests (remaining < 3)
        if self._last_remaining < 3 and self._last_remaining > 0:
            # We're critically low on remaining requests, add extra delay
            delay = self.MIN_REQUEST_DELAY + random.uniform(0.5, 1.0)
            logger.info(f"Critically low remaining ({self._last_remaining}), adding extra delay of {delay:.1f}s")
            time.sleep(delay)

    def _throttle(self) -> None:
        """
        Throttle requests to avoid hitting rate limits.
        Ensures minimum delay between consecutive requests.
        """
        self._wait_if_rate_limited()  # Check before making request

        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < self.MIN_REQUEST_DELAY:
            delay = self.MIN_REQUEST_DELAY + random.uniform(0, self.REQUEST_DELAY_JITTER)
            logger.debug(f"Throttling request by {delay:.2f}s")
            time.sleep(delay)

        self._last_request_time = time.time()
        self._request_count += 1

    def _update_rate_limit_state(self, response: requests.Response) -> None:
        """
        Update internal rate limit tracking from response headers.
        Only update reset time when we actually hit rate limits (remaining < 10).
        """
        remaining = int(response.headers.get("X-RateLimit-Remaining", 9999))
        reset_timestamp = int(response.headers.get("X-RateLimit-Reset", 0))

        self._last_remaining = remaining

        # Only update reset time if we're at risk of rate limiting AND have a valid reset time
        # We use separate tracking for different endpoint types to avoid cross-contamination
        # For search API vs content/repos API (they have independent rate limits)
        if remaining < 10 and reset_timestamp > time.time():
            # Only set reset time if it's in the future (avoid setting to past time)
            # And only if we have less than 5 remaining (more restrictive threshold)
            if remaining < 5:
                self._rate_limit_reset_at = max(reset_timestamp, time.time())
                logger.debug(f"Rate limit state updated: remaining={remaining}, reset_at={self._rate_limit_reset_at}")

    def _request(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        """
        Make an HTTP request with proper rate limit handling.
        Only ONE retry on rate limit - not multiple.
        """
        self._throttle()

        response = self.session.request(method, url, **kwargs)

        # Update rate limit tracking
        self._update_rate_limit_state(response)

        # Handle 403 Forbidden - rate limit or auth issue
        if response.status_code == 403:
            retry_after = int(response.headers.get("Retry-After", 0))
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            current_time = time.time()

            # Check if this is a rate limit (remaining=0) or auth issue
            # Rate limit: either retry-after > 0, or reset_time is in the future
            is_rate_limit = retry_after > 0 or (reset_time > current_time)

            if is_rate_limit:
                # This is a rate limit hit - wait and retry ONCE
                wait_seconds = retry_after if retry_after > 0 else max(reset_time - current_time, 0) + 1

                logger.warning(f"Rate limit hit (403). Waiting {wait_seconds}s before retry")
                self._rate_limit_reset_at = current_time + wait_seconds
                time.sleep(min(wait_seconds, 300))

                # Make ONE fresh request after waiting
                logger.info("Making fresh request after rate limit wait")
                response = self.session.request(method, url, **kwargs)
                self._update_rate_limit_state(response)

                # If still failing after retry, raise error - don't loop
                if response.status_code >= 400:
                    logger.error(f"Request still failing after retry: {response.status_code}")
                    response.raise_for_status()
            else:
                # This is an auth issue (no retry-after, not in rate limit window)
                logger.error(f"403 Forbidden (auth issue): {response.text[:200]}")
                response.raise_for_status()

        # Handle 429 Too Many Requests
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            current_time = time.time()
            wait_seconds = retry_after if retry_after > 0 else 60

            logger.warning(f"HTTP 429. Waiting {wait_seconds}s")
            self._rate_limit_reset_at = current_time + wait_seconds
            time.sleep(wait_seconds)

            # Make ONE fresh request after waiting
            response = self.session.request(method, url, **kwargs)
            self._update_rate_limit_state(response)

            if response.status_code >= 400:
                logger.error(f"Request still failing after 429 retry: {response.status_code}")
                response.raise_for_status()

        # Handle 502/503 Server Errors - these aren't rate limit issues
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
        Uses per_page=100 to maximize results per API call.
        """
        url = f"{self.BASE_URL}/search/code"
        params = {
            "q": query,
            "per_page": min(per_page, 100),  # Use 100 to get more results per call
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
        Uses per_page=100 to maximize results per API call.
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
                repo_query = f"{query} language:CUDA repo:{repo_name}"

                # Use per_page=100 (max allowed) to get more results per API call
                results = self.search_code(
                    query=repo_query,
                    per_page=100,  # Changed from min(30, remaining) to 100
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

                if len(items) < 100:
                    break

                page += 1
                delay = self.MIN_REQUEST_DELAY + random.uniform(0, self.REQUEST_DELAY_JITTER)
                time.sleep(delay)

            # Longer delay between repos
            delay = self.MIN_REQUEST_DELAY * 2 + random.uniform(0, 2)
            time.sleep(delay)

        all_items = self._sort_by_quality(all_items)
        return all_items[:max_results]

    def search_cuda_files_direct(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search for CUDA files using DIRECT GitHub code search.
        This is a simpler, faster approach that doesn't go through repos first.
        Uses per_page=100 to maximize results per API call.
        """
        all_items = []
        page = 1

        while len(all_items) < max_results:
            # Direct code search - no repo filtering
            results = self.search_code(
                query=query,
                per_page=100,
                page=page,
            )

            items = results.get("items", [])
            if not items:
                break

            all_items.extend(items)
            logger.info(f"Fetched {len(items)} items (total: {len(all_items)})")

            if len(items) < 100:
                break

            page += 1
            delay = self.MIN_REQUEST_DELAY + random.uniform(0, self.REQUEST_DELAY_JITTER)
            time.sleep(delay)

        all_items = self._sort_by_quality(all_items)
        return all_items[:max_results]

    def search_cuda_files_with_checkpoint(
        self,
        query: str,
        max_results: int = 100,
        checkpoint_data: dict[str, Any] | None = None,
        db_client=None,
        domain: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Search for CUDA files with database-backed pagination tracking.
        Uses per_page=100 to maximize results per API call.
        Persists pagination state to Neon DB for crash recovery and rate limit handling.

        Args:
            query: GitHub search query string
            max_results: Maximum results to return (up to 1000 = 10 pages)
            checkpoint_data: Legacy support - can be used if db_client not provided
            db_client: Database client for pagination state tracking (recommended)
            domain: Domain label for tracking (e.g., 'deep_learning')

        Returns:
            Tuple of (results_list, checkpoint_dict)
        """
        page = 1
        processed_count = 0
        seen_signatures = set()
        last_signature = ""

        # Check database for existing pagination state
        if db_client:
            progress = db_client.get_search_progress(query)
            if progress:
                if progress['status'] == 'failed':
                    # Check if rate limit has reset
                    from datetime import datetime
                    reset_time = progress.get('rate_limit_reset')
                    if reset_time and isinstance(reset_time, datetime):
                        if reset_time > datetime.now():
                            logger.info(f"Query {query} still rate limited until {reset_time}")
                            return [], {"query": query, "status": "rate_limited"}
                    # Rate limit expired, allow retry
                    logger.info(f"Query {query} rate limit expired, retrying from page {progress['current_page']}")

                if progress['status'] == 'completed':
                    logger.info(f"Query {query} already completed, skipping")
                    return [], {"query": query, "status": "completed"}

                if progress['last_result_count'] == 100:
                    # We stopped mid-search with more results available
                    page = progress['current_page'] + 1
                    processed_count = progress['total_processed']
                    logger.info(f"Resuming query {query} from page {page} (processed: {processed_count})")
                else:
                    # No more results available (last_result_count < 100 means end)
                    logger.info(f"Query {query} ended at page {progress['current_page']}")
                    return [], {"query": query, "status": "completed"}

        # Fallback to in-memory checkpoint if no db_client
        elif checkpoint_data:
            if checkpoint_data.get("query") == query:
                page = checkpoint_data.get("page", 1)
                processed_count = checkpoint_data.get("processed_count", 0)
                seen_signatures = set(checkpoint_data.get("seen_signatures", []))
                logger.info(f"Resuming (in-memory): page {page}, processed {processed_count}")
            else:
                logger.info("Query changed, starting fresh")

        all_items = []
        result_count = 0

        # Use direct code search - much faster than two-step repo->code
        while processed_count < max_results:
            try:
                results = self.search_code(
                    query=query,
                    per_page=100,
                    page=page,
                )

                items = results.get("items", [])
                result_count = len(items)

                if not items:
                    # No more results for this query
                    if db_client:
                        db_client.mark_search_completed(query)
                    break

                # Collect new items and signatures
                for item in items:
                    signature = (item.get("repository", {}).get("full_name"), item.get("path"))
                    if signature not in seen_signatures:
                        seen_signatures.add(signature)
                        all_items.append(item)
                        processed_count += 1

                # Update last signature from last item in batch
                if items:
                    last_item = items[-1]
                    last_signature = f"{last_item.get('repository', {}).get('full_name', '')}/{last_item.get('path', '')}"
                    logger.info(f"Page {page}: fetched {len(items)} items (total: {processed_count}, last: {last_signature[:60]}...)")

                # Update database progress after each page
                if db_client:
                    status = "in_progress"
                    if result_count < 100:
                        status = "completed"
                    db_client.upsert_search_progress(
                        query=query,
                        domain=domain,
                        current_page=page,
                        last_signature=last_signature,
                        last_result_count=result_count,
                        total_processed=processed_count,
                        status=status,
                    )

                # Check if we should continue pagination
                if result_count < 100:
                    # Less than full page = end of results
                    if db_client and result_count > 0:
                        db_client.mark_search_completed(query)
                    break

                if processed_count >= max_results:
                    break

                page += 1
                delay = self.MIN_REQUEST_DELAY + random.uniform(0, self.REQUEST_DELAY_JITTER)
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Search failed at page {page}: {e}")
                # Mark as failed with rate limit reset if applicable
                if db_client and "rate limit" in str(e).lower():
                    from datetime import datetime, timedelta
                    reset_time = datetime.now() + timedelta(minutes=5)
                    db_client.mark_search_failed(query, reset_time.isoformat())
                raise

        # Remove quality sorting to preserve query context diversity
        # all_items = self._sort_by_quality(all_items)

        checkpoint = {
            "query": query,
            "page": page,
            "processed_count": processed_count,
            "last_result_count": result_count,
            "seen_signatures": list(seen_signatures),
            "status": "completed" if result_count < 100 else "has_more",
        }

        return all_items[:max_results], checkpoint
