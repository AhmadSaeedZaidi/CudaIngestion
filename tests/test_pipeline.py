"""Tests for CUDA Kernel Ingestion Pipeline - Dry Run Mode.

These tests verify the pipeline operates correctly in dry-run mode,
ensuring NO calls are made to external APIs (MiniMax) or databases.
"""

import base64
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Sample CUDA kernel code for testing
SAMPLE_CUDA_KERNEL = '''
__global__ void matrixMultiply(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
'''


class TestCUDAFilter:
    """Test the CUDA filter heuristics."""

    def test_valid_kernel_passes_filter(self):
        """Test that a valid CUDA kernel passes the filter."""
        from src.processor.filter import CUDAFilter

        filter = CUDAFilter()
        passed, reason = filter.filter(SAMPLE_CUDA_KERNEL)

        assert passed, f"Valid kernel should pass filter: {reason}"

    def test_short_code_fails_filter(self):
        """Test that code that's too short fails the filter."""
        from src.processor.filter import CUDAFilter

        filter = CUDAFilter(min_length=100)
        passed, reason = filter.filter("__global__ void test() {}")

        assert not passed
        assert "too short" in reason.lower()

    def test_host_only_code_fails_filter(self):
        """Test that host-only code (no device keywords) fails."""
        from src.processor.filter import CUDAFilter

        filter = CUDAFilter()
        host_code = '''
void main() {
    std::cout << "Hello" << std::endl;
}
'''
        passed, reason = filter.filter(host_code)

        assert not passed

    def test_dummy_patterns_detected(self):
        """Test that dummy/test patterns are detected."""
        from src.processor.filter import CUDAFilter

        filter = CUDAFilter()
        test_code = '''
// test: implement this kernel
__global__ void dummy() {}
'''
        passed, reason = filter.filter(test_code)

        assert not passed


class TestQueryBuilder:
    """Test the query builder for domain diversity."""

    def test_query_generation(self):
        """Test that queries are generated correctly."""
        from src.scraper.query_builder import QueryBuilder

        builder = QueryBuilder()
        query = builder.get_next_query()

        assert "extension:cu" in query or "extension:cuh" in query

    def test_diverse_batch(self):
        """Test that diverse queries are generated."""
        from src.scraper.query_builder import QueryBuilder

        builder = QueryBuilder()
        queries = builder.get_diverse_batch(num_queries=5)

        assert len(queries) == 5
        assert len(set(queries)) == 5  # All unique

    def test_repo_filter_query(self):
        """Test that repo filter query is correctly generated."""
        from src.scraper.query_builder import QueryBuilder

        # Test default (quality bar: stars + non-forks)
        repo_query = QueryBuilder.get_repo_filter_query()
        assert "language:cuda" in repo_query
        assert "stars:>=100" in repo_query
        assert "fork:false" not in repo_query

        # Explicit lower star floor without fork filter
        repo_query_loose = QueryBuilder.get_repo_filter_query(min_stars=50, fork_filter=False)
        assert "stars:>=50" in repo_query_loose
        assert "fork:false" not in repo_query_loose

    def test_repo_discovery_queries_rotate(self):
        """Rotating repo queries should be distinct strings for long-running discovery."""
        from src.scraper.query_builder import QueryBuilder

        variants = QueryBuilder.repo_discovery_queries()
        assert len(variants) >= 3
        assert len(set(variants)) == len(variants)
        assert all("language:cuda" in v for v in variants)


class TestPipelineDryRun:
    """Test pipeline in dry-run mode with mocked external services."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.github_token = "test_token"
        config.minimax_api_key = "test_key"
        config.minimax_api_base = "https://api.test.com"
        config.neon_uri = "postgresql://localhost/test"
        config.batch_size = 10
        config.max_kernel_length = 50000
        config.min_kernel_length = 50
        config.dry_run = True
        config.repos_per_run = 50
        config.reset_github_repo_discovery = False
        return config

    def test_pipeline_initializes_in_dry_run(self, mock_config):
        """Test that pipeline initializes correctly in dry-run mode."""
        # Patch get_config before importing IngestionPipeline
        with patch.dict("sys.modules", {}):
            with patch("src.core.config.Config") as MockConfig:
                MockConfig.return_value = mock_config
                from src.main import IngestionPipeline

                # Create pipeline directly without calling get_config
                with patch("src.core.config.get_config", return_value=mock_config):
                    pipeline = IngestionPipeline(dry_run=True)
                    assert pipeline.dry_run is True




class TestPipelineIntegration:
    """Integration tests with extensive mocking."""

    def test_full_pipeline_dry_run(self):
        """Test full pipeline in dry-run mode."""
        # Setup mock config
        mock_config = Mock(
            github_token="test_token",
            minimax_api_key="test_key",
            minimax_api_base="https://api.test.com",
            neon_uri="postgresql://localhost/test",
            batch_size=10,
            max_kernel_length=50000,
            min_kernel_length=50,
            dry_run=True,
            repos_per_run=50,
            reset_github_repo_discovery=False,
        )

        # Mock GitHub response for code search
        encoded_content = base64.b64encode(SAMPLE_CUDA_KERNEL.encode()).decode()
        mock_code_search_result = {
            "items": [
                {
                    "repository": {"full_name": "test/repo"},
                    "path": "kernels/test.cu",
                    "sha": "abc123",
                }
            ]
        }
        mock_file_content = {"content": encoded_content}

        # Mock repo search result
        mock_repo_search_result = {
            "items": [
                {"full_name": "test/repo", "stargazers_count": 100},
                {"full_name": "another/repo", "stargazers_count": 50},
            ]
        }

        with patch("src.core.config.get_config", return_value=mock_config):
            with patch(
                "src.scraper.github_client.GitHubClient.search_repositories",
                return_value=mock_repo_search_result,
            ):
                with patch(
                    "src.scraper.github_client.GitHubClient.search_code",
                    return_value=mock_code_search_result,
                ):
                    with patch(
                        "src.scraper.github_client.GitHubClient.collect_cuda_hits_from_repos",
                        return_value=mock_code_search_result["items"],
                    ):
                        with patch(
                            "src.scraper.github_client.GitHubClient.get_file_content",
                            return_value=mock_file_content,
                        ):
                            with patch(
                                "src.scraper.github_client.GitHubClient.get_commits",
                                return_value=[{"sha": "abc123"}],
                            ):
                                with patch("src.main.DatabaseClient") as MockDBClient:
                                    # Setup DB Client mock
                                    mock_db = MockDBClient.return_value
                                    mock_db.get_next_repo_to_process.side_effect = [
                                        {"repo_name": "test/repo", "processed_page": 1, "last_commit_hash": "abc123", "available_kernels": 1, "explored_kernels": 0},
                                        None  # break the loop
                                    ]
                                    mock_db.check_duplicate.return_value = False

                                    from src.main import IngestionPipeline
                                    pipeline = IngestionPipeline(dry_run=True)

                                    # This should NOT raise any exceptions
                                    results = pipeline.run_batch(max_kernels=1)

                                    # In dry-run, no MiniMax annotation happens, but records pass filters
                                    # Note: "annotated" counts records that passed filters, not actual API calls
                                    assert results["annotated"] == 1
                                    assert results["filtered"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])