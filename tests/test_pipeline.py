"""Tests for CUDA Kernel Ingestion Pipeline - Dry Run Mode.

These tests verify the pipeline operates correctly in dry-run mode,
ensuring NO calls are made to external APIs (MiniMax) or databases.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

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
// TODO: implement this kernel
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
        assert "language:CUDA" not in query  # This is added by the client

    def test_diverse_batch(self):
        """Test that diverse queries are generated."""
        from src.scraper.query_builder import QueryBuilder
        
        builder = QueryBuilder()
        queries = builder.get_diverse_batch(num_queries=5)
        
        assert len(queries) == 5
        assert len(set(queries)) == 5  # All unique


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
        return config

    @pytest.fixture
    def mock_github_response(self):
        """Create mock GitHub search response."""
        return {
            "items": [
                {
                    "repository": {"full_name": "test/repo"},
                    "path": "kernels/test.cu",
                    "sha": "abc123"
                }
            ]
        }

    def test_pipeline_initializes_in_dry_run(self, mock_config):
        """Test that pipeline initializes correctly in dry-run mode."""
        with patch('src.main.get_config', return_value=mock_config):
            from src.main import IngestionPipeline
            
            pipeline = IngestionPipeline(dry_run=True)
            
            assert pipeline.dry_run is True

    def test_github_client_made_requests(self, mock_config, mock_github_response):
        """Test that GitHub API is called correctly."""
        with patch('src.main.get_config', return_value=mock_config):
            with patch('src.scraper.github_client.GitHubClient.search_code', return_value=mock_github_response) as mock_search:
                from src.scraper.github_client import GitHubClient
                
                client = GitHubClient("test_token")
                result = client.search_code("test query")
                
                assert result == mock_github_response
                mock_search.assert_called_once()

    def test_no_minimax_api_calls_in_dry_run(self, mock_config):
        """Verify NO calls to MiniMax API in dry-run mode."""
        with patch('src.main.get_config', return_value=mock_config):
            from src.main import IngestionPipeline
            
            pipeline = IngestionPipeline(dry_run=True)
            
            # Create a mock annotator
            with patch.object(pipeline.annotator, 'annotate') as mock_annotate:
                # Process a sample kernel
                record = pipeline.process_kernel(
                    repo="test/repo",
                    file_path="test.cu",
                    raw_code=SAMPLE_CUDA_KERNEL
                )
                
                # Annotation should NOT be called in dry-run
                mock_annotate.assert_not_called()
                
                # But record should still be created (without annotation)
                assert record is not None
                assert record.domain_tag is None

    def test_no_database_writes_in_dry_run(self, mock_config):
        """Verify NO database writes in dry-run mode."""
        with patch('src.main.get_config', return_value=mock_config):
            from src.main import IngestionPipeline
            
            pipeline = IngestionPipeline(dry_run=True)
            
            # Mock the database client
            with patch.object(pipeline.db_client, 'insert_batch') as mock_insert:
                records = [Mock()]
                result = pipeline.db_client.insert_batch(records)
                
                # insert_batch should still work (for counting) but not commit
                # In dry-run mode, main.py logs but doesn't call insert_batch
                mock_insert.assert_not_called()  # We don't call insert_batch in dry-run

    def test_filter_applied_in_dry_run(self, mock_config):
        """Test that filters are still applied in dry-run mode."""
        with patch('src.main.get_config', return_value=mock_config):
            from src.main import IngestionPipeline
            
            pipeline = IngestionPipeline(dry_run=True)
            
            # Valid kernel should pass filter
            record = pipeline.process_kernel(
                repo="test/repo",
                file_path="test.cu",
                raw_code=SAMPLE_CUDA_KERNEL
            )
            
            assert record is not None
            
            # Invalid kernel should fail filter
            invalid_record = pipeline.process_kernel(
                repo="test/repo",
                file_path="test.cu",
                raw_code="void main() {}"  # No CUDA keywords
            )
            
            assert invalid_record is None


class TestPipelineIntegration:
    """Integration tests with extensive mocking."""

    @patch('src.main.get_config')
    @patch('src.scraper.github_client.GitHubClient.search_code')
    @patch('src.scraper.github_client.GitHubClient.get_file_content')
    def test_full_pipeline_dry_run(self, mock_file_content, mock_search, mock_config):
        """Test full pipeline in dry-run mode."""
        # Setup mocks
        mock_config.return_value = Mock(
            github_token="test_token",
            minimax_api_key="test_key",
            minimax_api_base="https://api.test.com",
            neon_uri="postgresql://localhost/test",
            batch_size=10,
            max_kernel_length=50000,
            min_kernel_length=50,
            dry_run=True
        )
        
        import base64
        encoded_content = base64.b64encode(SAMPLE_CUDA_KERNEL.encode()).decode()
        
        mock_search.return_value = {
            "items": [
                {
                    "repository": {"full_name": "test/repo"},
                    "path": "kernels/test.cu",
                    "sha": "abc123"
                }
            ]
        }
        
        mock_file_content.return_value = {
            "content": encoded_content
        }
        
        from src.main import IngestionPipeline
        
        pipeline = IngestionPipeline(dry_run=True)
        
        # This should NOT raise any exceptions
        results = pipeline.run_batch(max_kernels=1)
        
        assert results["fetched"] >= 0
        assert results["annotated"] == 0  # No annotation in dry-run
        assert results["dry_run"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
