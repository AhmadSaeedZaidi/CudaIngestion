"""MiniMax M2.7 API integration for CUDA kernel annotation."""

import json
import time
from typing import Any

import requests
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.logger import get_logger

logger = get_logger(__name__)


class AnnotationSchema(BaseModel):
    """JSON schema for kernel annotation response."""
    domain_tag: str = Field(..., description="Computational domain (e.g., 'machine_learning', 'signal_processing')")
    algorithmic_intent: str = Field(..., description="2-3 sentence description of the algorithm's purpose")
    memory_pattern: str = Field(..., description="Memory access pattern (e.g., 'row-major', 'tiled', 'shared')")
    hardware_utilization: str = Field(..., description="Expected hardware utilization hints")
    mathematical_formulation: str = Field(..., description="The mathematical operation/formula implemented")
    thread_to_data_mapping: str = Field(..., description="How threadIdx/blockIdx maps to data indices")
    bottleneck_analysis: str = Field(..., description="GPU performance bottlenecks")
    edge_case_vulnerabilities: str = Field(..., description="Potential edge case issues")


class MiniMaxAnnotator:
    """
    MiniMax M2.7 API client for annotating CUDA kernels.
    Enforces strict JSON schema and implements exponential backoff.
    Supports batch processing to reduce API calls.
    """

    SYSTEM_PROMPT = """You are an expert in CUDA GPU programming. Analyze the provided CUDA kernel code and return a JSON object with the following schema:

{
  "domain_tag": "string (e.g., 'machine_learning', 'signal_processing', 'physics_simulation')",
  "algorithmic_intent": "string (2-3 sentences describing the algorithm's purpose)",
  "memory_pattern": "string (describe the memory access pattern: 'row-major', 'column-major', 'tiled', 'shared memory', 'coalesced', etc.)",
  "hardware_utilization": "string (describe expected GPU utilization hints: 'memory-bound', 'compute-bound', 'occupancy concerns', etc.)",
  "mathematical_formulation": "string (the core mathematical operation or formula this kernel computes, e.g., 'C[i,j] = sum(A[i,k]*B[k,j])', '1D FFT butterfly operation', 'N-body gravitational force sum')",
  "thread_to_data_mapping": "string (describe how threadIdx/blockIdx maps to data indices, e.g., 'thread computes element at row=blockIdx.y*blockDim.y+threadIdx.y, col=blockIdx.x*blockDim.x+threadIdx.x')",
  "bottleneck_analysis": "string (identify potential GPU bottlenecks: warp divergence, shared memory bank conflicts, uncoalesced global memory accesses, low occupancy, memory-bound vs compute-bound classification)",
  "edge_case_vulnerabilities": "string (describe potential edge case issues: boundary condition checks, race conditions in shared memory, out-of-bounds access risks, atomic operation conflicts)"
}

Return ONLY the JSON object, no additional text."""

    USER_PROMPT_TEMPLATE = """Analyze this CUDA kernel:

```cuda
{code}
```"""

    # Batch processing settings
    BATCH_SIZE = 10
    BATCH_DELAY = 2.0

    def __init__(self, api_key: str, api_base: str = "https://api.minimax.io/v1"):
        """
        Initialize MiniMax annotator.
        """
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    @retry(
        retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.Timeout, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=5, max=120),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _make_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Make API request with exponential backoff for rate limits.
        """
        url = f"{self.api_base}/text/chatcompletion_v2"
        response = self.session.post(url, json=payload, timeout=180)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning(f"Rate limited (429). Waiting {retry_after}s")
            time.sleep(retry_after)
            response.raise_for_status()

        if response.status_code == 502:
            logger.warning("Bad Gateway (502). Retrying with backoff")
            response.raise_for_status()

        response.raise_for_status()
        return response.json()

    def _parse_annotation(self, content: str) -> AnnotationSchema | None:
        """
        Parse JSON annotation from model response.
        """
        try:
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            data = json.loads(content)
            return AnnotationSchema(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse annotation JSON: {e}")
            return None

    def annotate(self, code: str) -> AnnotationSchema | None:
        """
        Annotate a single CUDA kernel.
        """
        max_code_length = 15000
        if len(code) > max_code_length:
            code = code[:max_code_length]

        user_prompt = self.USER_PROMPT_TEMPLATE.format(code=code)
        payload = {
            "model": "MiniMax-M2.7",
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }

        try:
            response = self._make_request(payload)
            choices = response.get("choices", [])
            if not choices:
                return None
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if not content:
                return None
            return self._parse_annotation(content)
        except Exception as e:
            logger.error(f"HTTP error during annotation: {e}")
            return None

    def _format_batch_prompt(self, codes: list[str]) -> str:
        """
        Format multiple kernels into a single prompt for batch annotation.
        """
        formatted = []
        for i, code in enumerate(codes, 1):
            max_len = 3000
            if len(code) > max_len:
                code = code[:max_len] + "..."
            formatted.append(f"=== Kernel {i} ===\n```{code}\n```")

        return f"""Analyze each of the following CUDA kernels and return a JSON array of annotations:

{chr(10).join(formatted)}

Return a JSON array with {len(codes)} annotation objects.

Return ONLY the JSON array, no additional text."""

    def _parse_batch_annotations(self, content: str, num_expected: int) -> list[AnnotationSchema | None]:
        """
        Parse batch annotation response.
        """
        try:
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            data = json.loads(content)
            if isinstance(data, list):
                annotations = []
                for item in data:
                    try:
                        annotations.append(AnnotationSchema(**item))
                    except (ValueError, TypeError):
                        annotations.append(None)
                return annotations
            else:
                return [None] * num_expected
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse batch annotation JSON: {e}")
            return [None] * num_expected

    def annotate_batch(self, codes: list[str]) -> list[AnnotationSchema | None]:
        """
        Annotate multiple kernels in batches to reduce API calls.
        """
        if not codes:
            return []

        results = []
        total_batches = (len(codes) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for batch_idx in range(total_batches):
            start = batch_idx * self.BATCH_SIZE
            end = min(start + self.BATCH_SIZE, len(codes))
            batch_codes = codes[start:end]

            logger.info(f"Annotating batch {batch_idx + 1}/{total_batches} ({len(batch_codes)} kernels)")

            # Truncate each code for batch prompt
            batch_codes = [code[:15000] if len(code) > 15000 else code for code in batch_codes]

            user_prompt = self._format_batch_prompt(batch_codes)
            payload = {
                "model": "MiniMax-M2.7",
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
            }

            try:
                response = self._make_request(payload)
                choices = response.get("choices", [])
                if not choices:
                    results.extend([None] * len(batch_codes))
                    continue
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if not content:
                    results.extend([None] * len(batch_codes))
                    continue
                batch_results = self._parse_batch_annotations(content, len(batch_codes))
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch annotation failed: {e}")
                results.extend([None] * len(batch_codes))

            # Delay between batches to respect rate limits
            if batch_idx < total_batches - 1:
                delay = self.BATCH_DELAY + (total_batches * 0.1)
                logger.debug(f"Waiting {delay:.1f}s before next batch")
                time.sleep(delay)

        return results
