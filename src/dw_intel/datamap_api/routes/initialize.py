from fastapi import APIRouter, Depends, HTTPException
from quack.datamap_api.models.config_request import ConfigRequest
from quack.shared.analyser import DataMapSchemaAnalyzer
from quack.datamap_api.deps.analyzer_lock import get_analyzer, set_analyzer
from quack.datamap_api.configuration.api import get_api_configuration

import json
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/initialize", tags=["root"])
async def initialize_analyzer(request: ConfigRequest):
    """Initialize the analyzer with provided configuration"""
    logger.info("Initializing analyzer")
    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    try:
        # Get current configuration
        config = get_api_configuration()
        
        # Update configuration with new values
        config.update(request.config)
        
        # Log the updated configuration
        logger.info("Updated configuration:")
        logger.info(f"S3 Configuration:")
        logger.info(f"  - Bucket: {config.bucket_name}")
        logger.info(f"  - Prefix: {config.prefix}")
        logger.info(f"  - Region: {config.aws_region}")
        logger.info(f"  - Pattern: {config.pattern}")
        logger.info(f"  - Cache Size: {config.cache_size}")
        
        logger.info(f"Bedrock Configuration:")
        logger.info(f"  - Region: {config.bedrock_region}")
        logger.info(f"  - Embeddings Model: {config.bedrock_embeddings_model}")
        logger.info(f"  - Inference Model: {config.bedrock_inference_model}")
        
        # Create new analyzer with updated configuration
        new_analyzer = DataMapSchemaAnalyzer(config)
        set_analyzer(new_analyzer)
        
        return {
            "message": "Analyzer initialized successfully",
            "configuration": {
                "s3": {
                    "bucket_name": config.bucket_name,
                    "prefix": config.prefix,
                    "region": config.aws_region,
                    "pattern": config.pattern,
                    "cache_size": config.cache_size
                },
                "bedrock": {
                    "region": config.bedrock_region,
                    "embeddings_model": config.bedrock_embeddings_model,
                    "inference_model": config.bedrock_inference_model
                }
            }
        }
    except Exception as e:
        logger.error(f"Error initializing analyzer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing analyzer: {str(e)}"
        )