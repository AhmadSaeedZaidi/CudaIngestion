"""Configuration management for CUDA Ingest Pipeline."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Neon PostgreSQL
    neon_uri: str | None = os.getenv("NEON_URI")

    # GitHub - use TOKEN_GITHUB (GitHub Actions restricts GITHUB_* prefix)
    github_token: str | None = os.getenv("TOKEN_GITHUB")

    # MiniMax M2.7 API
    minimax_api_key: str | None = os.getenv("MINIMAX_API_KEY")
    minimax_api_base: str = os.getenv("MINIMAX_API_BASE", "https://api.minimaxi.chat/v1")

    # Pipeline settings
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    max_kernel_length: int = int(os.getenv("MAX_KERNEL_LENGTH", "100000"))
    min_kernel_length: int = int(os.getenv("MIN_KERNEL_LENGTH", "50"))
    # Repo discovery page size (1–100). Lower for faster local runs (e.g. REPOS_PER_RUN=8).
    repos_per_run: int = int(os.getenv("REPOS_PER_RUN", "50"))

    # Dry run mode
    dry_run: bool = os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes")

    # Reset persisted GitHub repo pagination (page + query variant) to defaults on next run
    reset_github_repo_discovery: bool = os.getenv(
        "RESET_GITHUB_REPO_DISCOVERY", "false"
    ).lower() in ("true", "1", "yes")

    def validate(self) -> None:
        """Validate that required configuration is present."""
        if not self.neon_uri:
            raise ValueError("NEON_URI environment variable is required")
        if not self.github_token:
            raise ValueError("TOKEN_GITHUB environment variable is required (GITHUB_TOKEN not allowed in GitHub Actions)")
        if not self.minimax_api_key:
            raise ValueError("MINIMAX_API_KEY environment variable is required")


def get_config() -> Config:
    """Get a validated configuration instance."""
    config = Config()
    config.repos_per_run = max(1, min(100, config.repos_per_run))
    config.validate()
    return config
