"""MiniMax M2.7 API integration for CUDA kernel annotation."""

import json
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


class MiniMaxAnnotator:
    """
    MiniMax M2.7 API client for annotating CUDA kernels.
    Enforces strict JSON schema and implements exponential backoff.
    """

    SYSTEM_PROMPT = """You are an expert in CUDA GPU programming. Analyze the provided CUDA kernel code and return a JSON object with the following schema:

{
  "domain_tag": "string (e.g., 'machine_learning', 'signal_processing', 'physics_simulation')",
  "algorithmic_intent": "string (2-3 sentences describing the algorithm's purpose and computational approach)",
  "memory_pattern": "string (describe the memory access pattern: 'row-major', 'column-major', 'tiled', 'shared memory', 'coalesced', etc.)",
  "hardware_utilization": "string (describe expected GPU utilization hints: 'memory-bound', 'compute-bound', 'occupancy concerns', etc.)"
}

Return ONLY the JSON object, no additional text."""

    USER_PROMPT_TEMPLATE = """Analyze this CUDA kernel:

```cuda
{code}
```"""

    def __init__(self, api_key: str, api_base: str = "https://api.minimax.io/v1"):
        """
        Initialize MiniMax annotator.

        Args:
            api_key: MiniMax API key
            api_base: API base URL (default: international endpoint)
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

        Args:
            payload: Request payload

        Returns:
            JSON response

        Raises:
            requests.HTTPError: On HTTP errors after retries exhausted
        """
        url = f"{self.api_base}/text/chatcompletion_v2"

        # Handle 429 (Too Many Requests) and 502 (Bad Gateway) with extra backoff
        response = self.session.post(url, json=payload, timeout=120)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning(f"Rate limited (429). Waiting {retry_after}s before retry")
            import time
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

        Args:
            content: Raw response content

        Returns:
            Parsed AnnotationSchema or None on failure
        """
        try:
            # Try to extract JSON from the response
            content = content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])  # Remove ```json and ```

            # Parse JSON
            data = json.loads(content)
            return AnnotationSchema(**data)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse annotation JSON: {e}")
            logger.debug(f"Content: {content[:500]}")
            return None

    def annotate(self, code: str) -> AnnotationSchema | None:
        """
        Annotate a CUDA kernel with domain and algorithmic information.

        Args:
            code: Raw CUDA kernel code

        Returns:
            AnnotationSchema with domain_tag, algorithmic_intent, memory_pattern, hardware_utilization
        """
        # Truncate code if too long (M2.7 context limit)
        max_code_length = 15000
        if len(code) > max_code_length:
            logger.warning(f"Code truncated from {len(code)} to {max_code_length} chars")
            code = code[:max_code_length]

        user_prompt = self.USER_PROMPT_TEMPLATE.format(code=code)
        payload = {
            "model": "MiniMax-M2.7",
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,  # Low temperature for consistent JSON
        }

        try:
            response = self._make_request(payload)

            # Extract content from response
            choices = response.get("choices", [])
            if not choices:
                logger.error(f"No choices in response: {response}")
                return None

            message = choices[0].get("message", {})
            content = message.get("content", "")

            if not content:
                logger.error("Empty content in response")
                return None

            return self._parse_annotation(content)

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during annotation: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during annotation: {e}")
            return None

    def annotate_batch(self, codes: list[str]) -> list[AnnotationSchema | None]:
        """
        Annotate multiple kernels.

        Args:
            codes: List of CUDA kernel codes

        Returns:
            List of AnnotationSchema (or None for failures)
        """
        results = []
        for i, code in enumerate(codes):
            logger.info(f"Annotating kernel {i+1}/{len(codes)}")
            annotation = self.annotate(code)
            results.append(annotation)

            # Rate limiting between requests
            import time
            time.sleep(1)

        return results
