from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Dict, Any


class ApiConfiguration(BaseSettings):
    """
    Configuration for the API.
    """

    # S3 Configuration
    bucket_name: str = "project-dw_intel"
    prefix: str = "sbca/batch4/1299438/raw/"
    aws_region: str = "us-east-1"
    cache_size: int = 128
    pattern: str = ".*\\.parquet$"

    # Bedrock Configuration
    bedrock_region: str = "us-east-1"
    bedrock_embeddings_model: str = "amazon.titan-embed-text-v2:0"
    bedrock_inference_model: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    def update(self, new_config: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    class Config:
        env_prefix = "DATAMAP_"


@lru_cache(maxsize=1)
def get_api_configuration() -> ApiConfiguration:
    """
    Get the API configuration.
    """
    return ApiConfiguration()
