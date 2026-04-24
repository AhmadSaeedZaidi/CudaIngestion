"""Heuristic filters for CUDA kernel quality assurance."""

import re
from dataclasses import dataclass

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FilterResult:
    """Result of filtering a CUDA kernel."""
    passed: bool
    reason: str | None = None


class CUDAFilter:
    """
    Heuristic filters to ensure CUDA kernels meet quality thresholds.
    Filters out host-only wrappers, dummy code, and low-quality submissions.
    """

    # Required CUDA keywords for device code
    DEVICE_KEYWORDS = [
        "__global__",
        "__device__",
        "__shared__",
        "__device__",
        "cudaMalloc",
        "cudaMemcpy",
        "threadIdx",
        "blockIdx",
        "blockDim",
        "gridDim",
    ]

    # Patterns indicating dummy/test code
    DUMMY_PATTERNS = [
        r"^\s*//\s*(test|benchmark|demo|sample)",
        r"^\s*#include\s+\"fake",
        r"void\s+main\s*\(\s*\)",  # Host main in what should be device code
        r"cout\s*<<",  # C++ iostream in CUDA (often indicates test code)
        r"printf\s*\(\s*\"test",
        r"//\s*TODO",
        r"//\s*FIXME",
        r"//\s*HACK",
    ]

    # Patterns indicating actual kernel implementations
    KERNEL_PATTERNS = [
        r"__global__\s+void\s+\w+",  # Global kernel function
        r"__device__\s+",  # Device function
        r"cudaMalloc\s*\(",  # Memory allocation
        r"cudaMemcpy\s*\(",  # Memory copy
        r"<<<\s*\w+,\s*\w+",  # Kernel launch syntax
    ]

    # Minimum line count for a real kernel
    MIN_LINES = 10

    # Maximum ratio of comments to code (indicates commented-out code)
    MAX_COMMENT_RATIO = 0.7

    def __init__(self, min_length: int = 50, max_length: int = 50000):
        """
        Initialize filter with length constraints.

        Args:
            min_length: Minimum character length for a valid kernel
            max_length: Maximum character length for a valid kernel
        """
        self.min_length = min_length
        self.max_length = max_length

    def check_length(self, code: str) -> FilterResult:
        """
        Check if code length is within acceptable bounds.

        Args:
            code: Raw CUDA code

        Returns:
            FilterResult indicating pass/fail
        """
        length = len(code.strip())

        if length < self.min_length:
            return FilterResult(False, f"Code too short: {length} chars (min: {self.min_length})")

        if length > self.max_length:
            return FilterResult(False, f"Code too long: {length} chars (max: {self.max_length})")

        return FilterResult(True)

    def check_device_keywords(self, code: str) -> FilterResult:
        """
        Check if code contains required CUDA device keywords.

        Args:
            code: Raw CUDA code

        Returns:
            FilterResult indicating pass/fail
        """
        # Count how many device keywords are present
        keyword_count = sum(1 for kw in self.DEVICE_KEYWORDS if kw in code)

        # Require at least 2 device-related keywords
        if keyword_count < 2:
            return FilterResult(False, f"Insufficient device keywords: found {keyword_count}")

        return FilterResult(True)

    def check_kernel_patterns(self, code: str) -> FilterResult:
        """
        Check if code contains patterns indicating actual kernel implementations.

        Args:
            code: Raw CUDA code

        Returns:
            FilterResult indicating pass/fail
        """
        kernel_pattern_count = sum(1 for pattern in self.KERNEL_PATTERNS if re.search(pattern, code))

        if kernel_pattern_count == 0:
            return FilterResult(False, "No kernel patterns found (__global__, kernel launch, etc.)")

        return FilterResult(True)

    def check_dummy_patterns(self, code: str) -> FilterResult:
        """
        Check if code contains patterns indicating dummy or test code.

        Args:
            code: Raw CUDA code

        Returns:
            FilterResult indicating pass/fail
        """
        for pattern in self.DUMMY_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return FilterResult(False, f"Dummy/test pattern detected: {pattern}")

        return FilterResult(True)

    def check_comment_ratio(self, code: str) -> FilterResult:
        """
        Check if comment-to-code ratio is reasonable.

        Args:
            code: Raw CUDA code

        Returns:
            FilterResult indicating pass/fail
        """
        lines = code.split("\n")
        code_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("//")]
        comment_lines = [ln for ln in lines if ln.strip().startswith("//")]

        if len(code_lines) == 0:
            return FilterResult(False, "No actual code lines found")

        ratio = len(comment_lines) / len(code_lines)
        if ratio > self.MAX_COMMENT_RATIO:
            return FilterResult(False, f"Comment ratio too high: {ratio:.2f}")

        return FilterResult(True)

    def filter(self, code: str) -> tuple[bool, str]:
        """
        Run all filters on CUDA code.

        Args:
            code: Raw CUDA code

        Returns:
            Tuple of (passed, reason)
        """
        # Length check first (fast)
        result = self.check_length(code)
        if not result.passed:
            return False, result.reason

        # Check for device keywords
        result = self.check_device_keywords(code)
        if not result.passed:
            return False, result.reason

        # Check for kernel patterns
        result = self.check_kernel_patterns(code)
        if not result.passed:
            return False, result.reason

        # Check for dummy patterns
        result = self.check_dummy_patterns(code)
        if not result.passed:
            return False, result.reason

        # Check comment ratio
        result = self.check_comment_ratio(code)
        if not result.passed:
            return False, result.reason

        return True, "Passed all filters"
