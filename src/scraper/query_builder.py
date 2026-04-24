"""Query builder for diverse CUDA kernel domain coverage."""

import itertools
from collections.abc import Iterator


class QueryBuilder:
    """
    Builds GitHub search queries for CUDA files across diverse computational domains.
    Prevents mode collapse by cycling through different computational fields.

    Note: For code search (/search/code), only language:, repo:, path:, filename:,
    and extension: qualifiers are supported. Repository qualifiers like stars:>50
    and fork:false must be handled via the two-step repo-then-code approach.
    """

    # Domain-specific search terms to ensure diverse kernel coverage
    DOMAIN_TERMS = [
        # Machine Learning / Deep Learning
        ["deep learning", "neural network", "backpropagation", "gradient descent", "convolution"],
        # Robotics and Control
        ["robotics", "inverse kinematics", "path planning", "PID controller", "Kalman filter"],
        # Signal Processing
        ["signal processing", "FFT", "digital filter", "audio processing", "image filtering"],
        # Physics Simulations
        ["physics simulation", "fluid dynamics", "N-body", "particle system", "finite element"],
        # Scientific Computing
        ["scientific computing", "matrix multiplication", "sparse matrix", "eigenvalue", "linear solver"],
        # Computer Graphics
        ["ray tracing", "rasterization", "voxel", "physically based rendering", "shadow mapping"],
        # Cryptography
        ["cryptography", "hash function", "encryption", "AES", "elliptic curve"],
        # Bioinformatics
        ["bioinformatics", "sequence alignment", "DNA sequencing", "protein folding", "molecular dynamics"],
    ]

    # File extensions for CUDA
    EXTENSIONS = ["cu", "cuh"]

    def __init__(self, seed: int = 42):
        """
        Initialize query builder.

        Args:
            seed: Random seed for reproducible query ordering
        """
        self.seed = seed
        self._domain_cycle = self._create_domain_cycle()

    def _create_domain_cycle(self) -> Iterator[list[str]]:
        """
        Create an infinite cycle through domain terms.

        Yields:
            Lists of search terms for each domain
        """
        return itertools.cycle(self.DOMAIN_TERMS)

    def get_next_query(self) -> str:
        """
        Get the next search query with domain diversity.
        Note: Only includes qualifiers valid for /search/code endpoint.

        Returns:
            GitHub search query string for code search
        """
        domain_terms = next(self._domain_cycle)
        primary_term = domain_terms[0]

        # Build query with CUDA extension filter (valid for code search)
        query_parts = [
            f"{primary_term}",
            "extension:cu OR extension:cuh",
        ]

        return " ".join(query_parts)

    def build_query(
        self,
        domain: str,
    ) -> str:
        """
        Build a custom query for a specific domain.
        Note: Only includes qualifiers valid for /search/code endpoint.

        Args:
            domain: Domain term to search for

        Returns:
            GitHub search query string for code search
        """
        query_parts = [
            domain,
            "extension:cu OR extension:cuh",
        ]

        return " ".join(query_parts)

    def get_all_queries(self) -> list[str]:
        """
        Get all available domain queries for batch processing.

        Returns:
            List of query strings
        """
        queries = []
        for domain_terms in self.DOMAIN_TERMS:
            for term in domain_terms:
                queries.append(self.build_query(term))
        return queries

    def get_diverse_batch(self, num_queries: int = 5) -> list[str]:
        """
        Get a batch of diverse queries for parallel processing.

        Args:
            num_queries: Number of queries to return

        Returns:
            List of diverse query strings
        """
        return [next(self._domain_cycle)[0] for _ in range(num_queries)]

    @staticmethod
    def get_repo_filter_query(min_stars: int = 50, fork_filter: bool = False) -> str:
        """
        Get the repository search query for quality filtering.
        This is used with /search/repositories to find top CUDA repos.

        Args:
            min_stars: Minimum star count (default: 50)
            fork_filter: Whether to filter out forks (default: False)

        Returns:
            Repository search query string
        """
        query_parts = [
            "language:CUDA",
            f"stars:>{min_stars}",
        ]
        if fork_filter:
            query_parts.append("fork:false")

        return " ".join(query_parts)
