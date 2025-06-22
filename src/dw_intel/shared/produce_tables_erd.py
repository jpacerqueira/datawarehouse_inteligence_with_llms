import boto3
import json
import os
from typing import Dict, List, Any, Optional
import graphviz
from datetime import datetime
import logging
from dw_intel.shared.rag import BedrockDatamapRAG
from dw_intel.shared.s3_data_source import S3DataSource

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableRelationshipAnalyzer:
    def __init__(
        self,
        rag_instance: BedrockDatamapRAG,
        bucket_name: str,
        prefix: str = "",
        region_name: str = "us-east-1"
    ):
        """Initialize the TableRelationshipAnalyzer.
        
        Args:
            rag_instance (BedrockDatamapRAG): Instance of BedrockDatamapRAG with loaded schema cache
            bucket_name (str): S3 bucket name for storing ERD outputs
            region_name (str): AWS region name
        """
        self.rag = rag_instance
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.metadata_prefix = f"{prefix}_metadata/erd/"
        
    def _infer_relationships(self) -> Dict[str, Any]:
        """Analyze schema_cache_v2 to infer table relationships.
        
        Returns:
            Dict[str, Any]: Dictionary containing table relationships and metadata
        """
        relationships = {
            "tables": {},
            "relationships": [],
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_tables": 0,
                "total_relationships": 0
            }
        }
        
        # Process each table from schema_cache_v2
        for table_name, table_info in self.rag.schema_cache_v2.items():
            table_data = {
                "name": table_name,
                "columns": [],
                "primary_keys": [],
                "foreign_keys": [],
                "relationships": []
            }
            
            # Process columns
            for col in table_info.columns:
                column_info = {
                    "name": col.name,
                    "type": col.type,
                    "nullable": col.nullable,
                    "is_primary": col.is_primary,
                    "is_foreign": col.is_foreign,
                    "references": col.references,
                    "relationship_type": col.relationship_type
                }
                table_data["columns"].append(column_info)
                
                # Add primary key
                if col.is_primary:
                    table_data["primary_keys"].append(col.name)
                
                # Add foreign key
                if col.is_foreign and col.references:
                    fk_info = {
                        "column": col.name,
                        "references": col.references,
                        "relationship_type": col.relationship_type
                    }
                    table_data["foreign_keys"].append(fk_info)
                    
                    # Add relationship
                    relationships["relationships"].append({
                        "from_table": table_name,
                        "to_table": col.references["table"],
                        "from_column": col.name,
                        "to_column": col.references["column"],
                        "type": col.relationship_type or "many-to-one"
                    })
            
            relationships["tables"][table_name] = table_data
        
        # Update metadata
        relationships["metadata"]["total_tables"] = len(relationships["tables"])
        relationships["metadata"]["total_relationships"] = len(relationships["relationships"])
        
        return relationships
    
    def _generate_erd_dot(self, relationships: Dict[str, Any]) -> str:
        """Generate Graphviz DOT representation of the ERD.
        
        Args:
            relationships (Dict[str, Any]): Table relationships data
            
        Returns:
            str: DOT language representation of the ERD
        """
        dot = graphviz.Digraph(comment='Database ERD')
        dot.attr(rankdir='LR')
        
        # Add nodes (tables)
        for table_name, table_data in relationships["tables"].items():
            # Create table label with columns
            table_label = f"{table_name}|"
            
            # Add primary keys first
            for pk in table_data["primary_keys"]:
                table_label += f"+ {pk} (PK)\\l"
            
            # Add foreign keys
            for fk in table_data["foreign_keys"]:
                table_label += f"# {fk['column']} (FK → {fk['references']['table']}.{fk['references']['column']})\\l"
            
            # Add other columns
            for col in table_data["columns"]:
                if col["name"] not in table_data["primary_keys"] and not any(fk["column"] == col["name"] for fk in table_data["foreign_keys"]):
                    table_label += f"  {col['name']} : {col['type']}\\l"
            
            dot.node(table_name, f"{{{table_label}}}", shape="record")
        
        # Add edges (relationships)
        for rel in relationships["relationships"]:
            # Determine arrow style based on relationship type
            arrowhead = "crow" if rel["type"] == "many-to-one" else "normal"
            arrowtail = "crow" if rel["type"] == "one-to-many" else "normal"
            
            dot.edge(
                rel["from_table"],
                rel["to_table"],
                label=f"{rel['from_column']} → {rel['to_column']}",
                dir="both",
                arrowhead=arrowhead,
                arrowtail=arrowtail
            )
        
        return dot
    
    def generate_and_store_erd(self) -> Dict[str, str]:
        """Generate ERD and store in S3.
        
        Returns:
            Dict[str, str]: Dictionary containing S3 paths to generated files
        """
        try:
            # Generate relationships data
            relationships = self._infer_relationships()
            
            # Store relationships JSON
            json_key = f"{self.metadata_prefix}relationships_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=json_key,
                Body=json.dumps(relationships, indent=2),
                ContentType='application/json'
            )
            
            result = {"json": json_key}
            
            try:
                # Try to generate visual ERD
                dot = self._generate_erd_dot(relationships)
                
                # Store SVG
                svg_key = f"{self.metadata_prefix}erd_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.svg"
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=svg_key,
                    Body=dot.pipe(format='svg'),
                    ContentType='image/svg+xml'
                )
                result["svg"] = svg_key
                
                # Store PNG
                png_key = f"{self.metadata_prefix}erd_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=png_key,
                    Body=dot.pipe(format='png'),
                    ContentType='image/png'
                )
                result["png"] = png_key
                
                logger.info(f"Successfully generated and stored ERD files in {self.metadata_prefix}")
                
            except Exception as viz_error:
                logger.warning(f"Could not generate visual ERD diagrams: {str(viz_error)}")
                logger.warning("Only JSON relationships will be available. Install Graphviz for visual diagrams.")
                logger.warning("To install Graphviz:")
                logger.warning("  - On macOS: brew install graphviz")
                logger.warning("  - On Ubuntu/Debian: apt-get install graphviz")
                logger.warning("  - On Windows: Download from https://graphviz.org/download/")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating ERD: {str(e)}")
            raise
    
    def get_latest_erd_files(self) -> Dict[str, str]:
        """Get the most recent ERD files from S3.
        
        Returns:
            Dict[str, str]: Dictionary containing S3 paths to latest files
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.metadata_prefix
            )
            
            latest_files = {
                "json": None,
                "svg": None,
                "png": None
            }
            
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.json'):
                    latest_files["json"] = key
                elif key.endswith('.svg'):
                    latest_files["svg"] = key
                elif key.endswith('.png'):
                    latest_files["png"] = key
            
            return latest_files
            
        except Exception as e:
            logger.error(f"Error getting latest ERD files: {str(e)}")
            raise 
