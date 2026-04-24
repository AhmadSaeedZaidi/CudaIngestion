"""Query builder for diverse CUDA kernel domain coverage."""

import itertools
from collections.abc import Iterator


class QueryBuilder:
    """
    Builds GitHub search queries for CUDA files across diverse computational domains.
    Prevents mode collapse by cycling through different computational fields.
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

    # Minimum stars to filter for quality (can be adjusted)
    MIN_STARS = 0

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

        Returns:
            GitHub search query string
        """
        domain_terms = next(self._domain_cycle)
        # Pick one term from the domain randomly (simplified - uses round-robin)
        primary_term = domain_terms[0]

        query_parts = [
            f"{primary_term}",
            "extension:cu OR extension:cuh",
        ]

        return " ".join(query_parts)

    def build_query(
        self,
        domain: str,
        include_stars: bool = False,
        min_stars: int = 0,
    ) -> str:
        """
        Build a custom query for a specific domain.

        Args:
            domain: Domain term to search for
            include_stars: Whether to filter by stars
            min_stars: Minimum star count if filtering

        Returns:
            GitHub search query string
        """
        query_parts = [
            domain,
            "extension:cu OR extension:cuh",
        ]

        if include_stars and min_stars > 0:
            query_parts.append(f"stars:>={min_stars}")

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
