from typing import List, Dict, Any, Tuple, Optional
import logging
import time
import json

import streamlit as st
import pandas as pd
from dw_intel.shared.rag import BedrockDatamapRAG
from dw_intel.shared.sql_plan_validator import SQLPlanValidator
from dw_intel.shared.produce_tables_erd import TableRelationshipAnalyzer
from dw_intel.shared.duckdb_adapter import (
    execute_s3_pattern_query,
    get_duckdb_s3_connection,
)

logger = logging.getLogger(__name__)


class DataMapSchemaAnalyzer:
    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        region_name: str = "us-east-1",
        cache_size: int = 128,
    ):
        """Initialize the DataMapSchemaAnalyzer with S3 configuration.

        Args:
            bucket_name (str): S3 bucket name containing parquet files
            prefix (str): Prefix for parquet files in the bucket
            region_name (str): AWS region name
            cache_size (int): Maximum number of files to cache in memory
        """
        self.rag = BedrockDatamapRAG(
            bucket_name=bucket_name,
            prefix=prefix,
            region_name=region_name,
            embeddings_model="amazon.titan-embed-text-v2:0",
            inference_model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            cache_size=cache_size,
        )
        self._initialize_rag()
        # Initialize SQL plan validator
        self.sql_validator = SQLPlanValidator(self.rag.schema_cache)
        # Initialize ERD analyzer
        self.erd_analyzer = TableRelationshipAnalyzer(
            rag_instance=self.rag,
            bucket_name=bucket_name,
            prefix=prefix,
            region_name=region_name
        )

    def _initialize_rag(self, pattern: Optional[str] = None):
        """Initialize the RAG system with the S3 schema or hydration_data."""
        self.rag.build_rag_index(pattern)
        self.rag.build_rag_index_v2(pattern)
        # build with hydration_data
        self.rag.build_rag_index_v3(pattern)

    def analyze_schema(
        self,
        query: str,
        context: str = "cashflow",
        format_type: str = "sql",
    ) -> Dict[str, Any]:
        """Analyze the database schema based on a natural language query."""
        analysis = self.rag.get_detailed_schema_analysis(query, context, format_type)
        return {"analysis": analysis, "query": query}

    def get_sql_in_context(
        self,
        query: str,
        context: str = "cashflow",
        format_type: str = "sql",
    ) -> Dict[str, Any]:
        """Get SQL in context based on the query. with validation status and message"""

        sql_in_context = self.rag.get_detailed_sql_in_context(
            query,
            context,
            format_type,
        )

        # Clean the SQL plan
        sql_in_context = self.sql_validator.clean_sql_plan(sql_in_context)

        # Validate the SQL plan
        is_valid, validation_message = self.sql_validator.validate_sql_plan(
            sql_in_context, context
        )
        if not is_valid:
            logger.warning(f"SQL validation failed: {validation_message}")
            return {
                "sql_in_context": sql_in_context,
                "query": query,
                "validation_status": "failed",
                "validation_message": validation_message,
            }

        return {
            "sql_in_context": sql_in_context,
            "query": query,
            "validation_status": "success",
            "validation_message": "SQL plan validated successfully",
        }

    def get_sql_in_context_v2(
        self,
        query: str,
        context: str = "cashflow",
        format_type: str = "sql",
    ) -> Dict[str, Any]:
        """Get SQL in context V2 based on the query. Embeddings from hydration_data, with validation status and message."""

        sql_in_context = self.rag.get_detailed_sql_in_context_v2(
            query,
            context,
            format_type,
        )

        # Clean the SQL plan
        sql_in_context = self.sql_validator.clean_sql_plan(sql_in_context)

        # Validate the SQL plan
        is_valid, validation_message = self.sql_validator.validate_sql_plan(
            sql_in_context, context
        )
        if not is_valid:
            logger.warning(f"SQL validation failed: {validation_message}")
            return {
                "sql_in_context": sql_in_context,
                "query": query,
                "validation_status": "failed",
                "validation_message": validation_message,
            }

        return {
            "sql_in_context": sql_in_context,
            "query": query,
            "validation_status": "success",
            "validation_message": "SQL plan validated successfully",
        }

    def get_similar_schema(
        self, query: str, k: int = 3
    ) -> Tuple[List[str], List[float]]:
        """Get similar schema entries with similarity scores."""
        return self.rag.query_schema(query, k)

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get detailed schema information about a specific table."""
        return self.rag.get_table_schema(table_name)

    def get_column_info(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific column."""
        return self.rag.get_column_info(table_name, column_name)

    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire database schema."""
        all_tables = self.rag.schema_cache
        summary = {"total_tables": len(all_tables), "tables": []}

        for table_name, schema in all_tables.items():
            table_info = {
                "name": table_name,
                "column_count": len(schema["columns"]),
                "columns": [col["name"] for col in schema["columns"]],
                "row_count": schema["row_count"],
                "last_modified": schema["last_modified"],
                "size_bytes": schema["size_bytes"],
            }
            summary["tables"].append(table_info)

        return summary

    def generate_erd(self, query: str = "Analyze all tables and their relationships", context: str = "database_schema", format_type: str = "json") -> Dict[str, str]:
        """Generate ERD and return file paths."""
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
                
            # First get enriched schema using the provided query
            enriched_schema = self.rag.get_detailed_erd_schema(
                query=query,
                context=context,
                format_type=format_type)
            
            if not enriched_schema:
                raise ValueError("Failed to generate enriched schema")
            
            # Generate and store ERD
            erd_files = self.erd_analyzer.generate_and_store_erd()
            
            if not erd_files:
                raise ValueError("Failed to generate ERD files")
            
            return enriched_schema, erd_files
        except Exception as e:
            logger.error(f"Error generating ERD: {str(e)}")
            raise


def show_loading_message(message: str, max_dots: int = 3):
    """Show a loading message with augmenting dots."""
    placeholder = st.empty()
    dots = 0
    while True:
        dots = (dots + 1) % (max_dots + 1)
        loading_text = message + "." * dots
        placeholder.info(loading_text)
        time.sleep(0.5)
        yield loading_text


st.title("Cashflow DataMap Schema Analyzer")

# Initialize session state
if "analyzer" not in st.session_state:
    st.session_state.analyzer = None

# Initialize global configuration variables
if "s3_config" not in st.session_state:
    st.session_state.s3_config = {
        "bucket_name": "project-dw_intel",
        "prefix": "sbca/batch4/1299438/raw/",
        "region_name": "us-east-1",
        "cache_size": 128,
        "pattern": ".*\\.parquet$",
    }

# S3 configuration
col1, col2 = st.columns(2)
with col1:
    st.session_state.s3_config["bucket_name"] = st.text_input(
        "S3 Bucket Name:", value=st.session_state.s3_config["bucket_name"]
    )
with col2:
    st.session_state.s3_config["prefix"] = st.text_input(
        "S3 Prefix:", value=st.session_state.s3_config["prefix"]
    )

# Advanced settings
with st.expander("Advanced Settings"):
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.s3_config["region_name"] = st.text_input(
            "AWS Region:", value=st.session_state.s3_config["region_name"]
        )
    with col2:
        st.session_state.s3_config["cache_size"] = st.number_input(
            "Cache Size:",
            min_value=1,
            max_value=1000,
            value=st.session_state.s3_config["cache_size"],
        )
    st.session_state.s3_config["pattern"] = st.text_input(
        "File Pattern (optional):", value=st.session_state.s3_config["pattern"]
    )

if st.button("Initialize Analyzer"):
    try:
        with st.spinner("Initializing schema analyzer..."):
            st.session_state.analyzer = DataMapSchemaAnalyzer(
                bucket_name=st.session_state.s3_config["bucket_name"],
                prefix=st.session_state.s3_config["prefix"],
                region_name=st.session_state.s3_config["region_name"],
                cache_size=st.session_state.s3_config["cache_size"],
            )
            st.session_state.analyzer._initialize_rag(
                st.session_state.s3_config["pattern"]
            )
            st.success("Analyzer initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing analyzer: {str(e)}")

if st.session_state.analyzer:
    # Query input
    query = st.text_input("Enter your query about the database schema:")

    # Context and format type selection
    col1, col2 = st.columns(2)
    with col1:
        context = st.selectbox(
            "Context:",
            [
                "Forecast Cashflow",
                "Sales and Inventory",
                "Plan Liquidity",
                "Detect Cashflow Risks",
            ],
            index=0,
        )
    with col2:
        format_type = st.selectbox(
            "Output Format:",
            [
                "DuckDB SQL extended with PostgreSQL syntax",
                "DuckDB SQL extended with SQLite syntax",
                "PostgresSQL",
                "SQL",
            ],
            index=0,
        )

    if query and query.strip():
        try:
            # Create tabs for different functionalities
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                [
                    "Schema Analysis",
                    "SQL Generation",
                    "SQL Generation V2",
                    "Similar Schema",
                    "Schema Summary",
                    "Execute SQL",
                    "ERD Visualization"
                ]
            )

            with tab1:
                if not "analyzed_schema" in st.session_state:
                    st.session_state.analysis = (
                        st.session_state.analyzer.analyze_schema(
                            query,
                            context,
                            format_type,
                        )
                    )

                st.subheader("Schema Analysis")
                st.write("Analysis with Data Profiling of tables in RAG embeddings")
                st.write(st.session_state.analysis["analysis"])

            with tab2:
                st.subheader("SQL Generation")
                if "sql_result" not in st.session_state:
                    st.session_state.sql_result = st.session_state.analyzer.get_sql_in_context(
                        query,
                        context,
                        format_type,
                    )
                
                st.write("Generated SQL:")
                st.code(st.session_state.sql_result["sql_in_context"], language="sql")

                # Display validation status and message
                validation_status = st.session_state.sql_result.get("validation_status", "unknown")
                validation_message = st.session_state.sql_result.get(
                    "validation_message", "No validation message available"
                )

                if validation_status == "success":
                    st.success(f"Validation Status: {validation_status}")
                    st.info(f"Validation Message: {validation_message}")
                else:
                    st.error(f"Validation Status: {validation_status}")
                    st.warning(f"Validation Message: {validation_message}")

            with tab3:
                st.subheader("SQL Generation V2 - Human in the loop - Data Hydration")
                if "sql_result_v2" not in st.session_state:
                    st.session_state.sql_result_v2 = st.session_state.analyzer.get_sql_in_context_v2(
                        query,
                        context,
                        format_type,
                    )
                
                st.write("Generated SQL V2:")
                st.code(st.session_state.sql_result_v2["sql_in_context"], language="sql")

                # Display validation status and message
                validation_status = st.session_state.sql_result_v2.get("validation_status", "unknown")
                validation_message = st.session_state.sql_result_v2.get(
                    "validation_message", "No validation message available"
                )

                if validation_status == "success":
                    st.success(f"Validation Status: {validation_status}")
                    st.info(f"Validation Message: {validation_message}")
                else:
                    st.error(f"Validation Status: {validation_status}")
                    st.warning(f"Validation Message: {validation_message}")

            with tab4:
                st.subheader("Similar Schema")
                st.write("Similar schema entries with similarity scores")
                similar_schema, scores = st.session_state.analyzer.get_similar_schema(
                    query
                )
                for i, (text, score) in enumerate(zip(similar_schema, scores)):
                    st.write(f"Result {i+1} (Score: {score:.4f}):")
                    st.write(text)

            with tab5:
                st.subheader("Schema Summary")
                st.write("Data Profiling of tables and columns")
                summary = st.session_state.analyzer.get_schema_summary()
                st.write(f"Total Tables: {summary['total_tables']}")

                for table in summary["tables"]:
                    with st.expander(f"Table: {table['name']}"):
                        st.write(f"Columns ({table['column_count']}):")
                        st.write(table["columns"])
                        st.write(f"Row Count: {table['row_count']}")
                        st.write(f"Last Modified: {table['last_modified']}")
                        st.write(f"Size: {table['size_bytes']} bytes")

                        # Show table schema
                        table_schema = st.session_state.analyzer.get_table_schema(
                            table["name"]
                        )
                        if table_schema:
                            st.write("Column Details:")
                            df = pd.DataFrame(table_schema["columns"])
                            st.dataframe(df)

                            # Column analysis
                            selected_column = st.selectbox(
                                "Select a column for detailed analysis:",
                                [col["name"] for col in table_schema["columns"]],
                                key=f"col_select_{table['name']}",
                            )

                            if selected_column:
                                column_info = st.session_state.analyzer.get_column_info(
                                    table["name"], selected_column
                                )
                                st.write("Column Analysis:")
                                st.write(f"Type: {column_info['type']}")
                                st.write(f"Nullable: {column_info['nullable']}")
                                st.write(
                                    f"Unique Values: {column_info['unique_values']}"
                                )
                                st.write(f"Null Count: {column_info['null_count']}")
                                st.write("Sample Values:")
                                st.write(column_info["sample_values"])

            with tab6:
                st.subheader("Execute SQL")

                # Initialize SQL version in session state if not present
                if "sql_version" not in st.session_state:
                    st.session_state.sql_version = "Original SQL"

                # Add radio button for SQL version selection
                sql_version = st.radio(
                    "Select SQL Version to Execute:",
                    ["Original SQL", "Enhanced SQL (v2)"],
                    help="Choose between the original SQL generation or the enhanced version with better context understanding",
                    key="sql_version_radio"
                )

                # Update session state if version changed
                if sql_version != st.session_state.sql_version:
                    st.session_state.sql_version = sql_version

                # Select SQL based on user choice
                if st.session_state.sql_version == "Original SQL":
                    if "sql_result" in st.session_state:
                        sql_to_execute = st.session_state.sql_result["sql_in_context"]
                        st.info("Using original SQL generation")
                    else:
                        st.warning("Please generate SQL in the 'SQL Generation' tab first.")
                        sql_to_execute = None
                else:
                    if "sql_result_v2" in st.session_state:
                        sql_to_execute = st.session_state.sql_result_v2["sql_in_context"]
                        st.info("Using enhanced SQL generation (v2)")
                    else:
                        st.warning("Please generate SQL in the 'SQL Generation V2' tab first.")
                        sql_to_execute = None

                if sql_to_execute:
                    # S3 configuration inputs
                    st.info(
                        f"""Using S3 location to mount litle DW and execute sql query on RAG profiled tables"""
                    )

                    # Use the S3 configuration
                    s3_path = f"s3://{st.session_state.s3_config['bucket_name']}/{st.session_state.s3_config['prefix']}"
                    pattern = st.session_state.s3_config["pattern"]
                    region = st.session_state.s3_config["region_name"]
                    
                    st.code(sql_to_execute, language="sql")
                    st.info(
                        f"""SQL query that is validated by SQLPlanValidator tasks"""
                    )

                    # Initialize execution state if not present
                    if "execution_state" not in st.session_state:
                        st.session_state.execution_state = {
                            "is_executing": False,
                            "results": None,
                            "error": None
                        }

                    if st.button("Execute Query"):
                        # Set execution state
                        st.session_state.execution_state["is_executing"] = True
                        st.session_state.execution_state["results"] = None
                        st.session_state.execution_state["error"] = None

                        # Create a loading message generator
                        loading_gen = show_loading_message("Executing query")

                        try:
                            # Get connection with loading message
                            next(loading_gen)
                            local_connection = get_duckdb_s3_connection(region=region)

                            # log the connection
                            logger.info(f"Connection: {local_connection}")
                            st.info("Connection established")

                            # Execute query with loading message
                            next(loading_gen)
                            results = execute_s3_pattern_query(
                                query=sql_to_execute,
                                s3_path=s3_path,
                                pattern=pattern,
                                connection=local_connection,
                            )
                            # log the results
                            logger.info(f"Results: {results}")
                            st.info("Query executed")

                            # Store results in session state
                            st.session_state.execution_state["results"] = results
                            st.session_state.execution_state["is_executing"] = False

                            # Clear loading message
                            st.empty()

                            # Display results
                            st.subheader("Query Results")
                            for table_name, result in results.items():
                                with st.expander(f"Results for {table_name}"):
                                    # Convert result to DataFrame for display
                                    df = result.df()
                                    st.dataframe(df)

                                    # Show query statistics
                                    st.write(f"Number of rows: {len(df)}")
                                    st.write(f"Number of columns: {len(df.columns)}")

                                    # Download button
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        label="Download as CSV",
                                        data=csv,
                                        file_name=f"{table_name}_results.csv",
                                        mime="text/csv",
                                    )

                        except Exception as e:
                            # Clear loading message
                            st.empty()
                            st.session_state.execution_state["error"] = str(e)
                            st.session_state.execution_state["is_executing"] = False
                            st.error(f"Error executing query: {str(e)}")
                            st.error(
                                "Please check your S3 configuration and try again."
                            )
                    else:
                        # Display previous results if they exist
                        if st.session_state.execution_state["results"]:
                            st.subheader("Previous Query Results")
                            for table_name, result in st.session_state.execution_state["results"].items():
                                with st.expander(f"Results for {table_name}"):
                                    df = result.df()
                                    st.dataframe(df)
                                    st.write(f"Number of rows: {len(df)}")
                                    st.write(f"Number of columns: {len(df.columns)}")
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        label="Download as CSV",
                                        data=csv,
                                        file_name=f"{table_name}_results.csv",
                                        mime="text/csv",
                                    )

            with tab7:
                st.subheader("Entity Relationship Diagram (ERD)")
                
                # Initialize ERD query state with default value if not exists
                if "erd_query" not in st.session_state:
                    st.session_state.erd_query = "Analyze all relationships between all tables, their primary and foreign keys too for all tables"
                
                # Add modifiable query input with the default value
                modified_query = st.text_area(
                    "Modify the query for ERD analysis",
                    value=st.session_state.erd_query,
                    height=100,
                    key="erd_query_editor"
                )
                
                st.subheader("ERD Analysis Query")
                st.write(modified_query)
                
                # First get the LLM's schema analysis
                try:
                    with st.spinner("Analyzing schema relationships..."):
                        # Add a button to proceed with ERD generation
                        if st.button("Generate ERD Visualization"):
                            try:
                                with st.spinner("Generating ERD visualization..."):
                                    # Update the ERD query state
                                    st.session_state.erd_query = modified_query
                                    
                                    # Validate query
                                    if not modified_query or not modified_query.strip():
                                        st.error("Please enter a valid query for ERD analysis")
                                        st.stop()
                                    
                                    # Generate ERD with modified query
                                    enriched_schema, erd_files = st.session_state.analyzer.generate_erd(
                                        query=modified_query,
                                        context="database_schema",
                                        format_type="json"
                                    )

                                    # Display the enriched schema analysis
                                    st.subheader("Schema Relationship Analysis")
                                    st.json(enriched_schema)
                                    
                                    # Get the latest ERD files
                                    latest_files = st.session_state.analyzer.erd_analyzer.get_latest_erd_files()
                                    
                                    # Display ERD visualization
                                    if latest_files["svg"]:
                                        st.subheader("ERD Visualization")
                                        try:
                                            # Fetch SVG content from S3
                                            svg_response = st.session_state.analyzer.rag.s3_data_source.s3_client.get_object(
                                                Bucket=st.session_state.s3_config["bucket_name"],
                                                Key=latest_files["svg"]
                                            )
                                            svg_content = svg_response['Body'].read().decode('utf-8')
                                            st.image(svg_content, caption="Entity Relationship Diagram")
                                        except Exception as e:
                                            st.error(f"Error loading SVG visualization: {str(e)}")
                                    
                                    # Display relationship data
                                    if latest_files["json"]:
                                        st.subheader("Relationship Data")
                                        try:
                                            # Read and display JSON data
                                            json_response = st.session_state.analyzer.rag.s3_data_source.s3_client.get_object(
                                                Bucket=st.session_state.s3_config["bucket_name"],
                                                Key=latest_files["json"]
                                            )
                                            relationship_data = json.loads(json_response['Body'].read().decode('utf-8'))
                                            
                                            # Display tables and their relationships
                                            for table_name, table_info in relationship_data["tables"].items():
                                                with st.expander(f"Table: {table_name}"):
                                                    # Display primary keys
                                                    if table_info["primary_keys"]:
                                                        st.write("Primary Keys:")
                                                        st.write(table_info["primary_keys"])
                                                    
                                                    # Display foreign keys
                                                    if table_info["foreign_keys"]:
                                                        st.write("Foreign Keys:")
                                                        for fk in table_info["foreign_keys"]:
                                                            st.write(f"- {fk['column']} → {fk['references']['table']}.{fk['references']['column']}")
                                                    
                                                    # Display columns with enhanced formatting
                                                    st.write("Columns:")
                                                    # Create a DataFrame with enhanced column information
                                                    columns_data = []
                                                    for col in table_info["columns"]:
                                                        col_info = {
                                                            "Name": col["name"],
                                                            "Type": col["type"],
                                                            "Nullable": "Yes" if col["nullable"] else "No",
                                                            "Primary Key": "Yes" if col["name"] in table_info["primary_keys"] else "No",
                                                            "Foreign Key": "Yes" if any(fk["column"] == col["name"] for fk in table_info["foreign_keys"]) else "No",
                                                            "References": f"{fk['references']['table']}.{fk['references']['column']}" if any(fk["column"] == col["name"] for fk in table_info["foreign_keys"]) else "-"
                                                        }
                                                        columns_data.append(col_info)
                                                    
                                                    df = pd.DataFrame(columns_data)
                                                    
                                                    # Style the DataFrame
                                                    def highlight_keys(val):
                                                        if val == "Yes":
                                                            return 'background-color: #90EE90'  # Light green for Yes
                                                        return ''
                                                    
                                                    styled_df = df.style.applymap(highlight_keys, subset=['Primary Key', 'Foreign Key'])
                                                    st.dataframe(styled_df)
                                            
                                            # Display relationships
                                            st.subheader("Table Relationships")
                                            for rel in relationship_data["relationships"]:
                                                st.write(f"{rel['from_table']} → {rel['to_table']} ({rel['type']})")
                                                st.write(f"  {rel['from_column']} → {rel['to_column']}")
                                            
                                            # Add download buttons
                                            st.subheader("Download ERD")
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                try:
                                                    # Fetch SVG file from S3
                                                    svg_response = st.session_state.analyzer.rag.s3_data_source.s3_client.get_object(
                                                        Bucket=st.session_state.s3_config["bucket_name"],
                                                        Key=latest_files["svg"]
                                                    )
                                                    svg_data = svg_response['Body'].read()
                                                    st.download_button(
                                                        "Download SVG",
                                                        data=svg_data,
                                                        file_name="erd.svg",
                                                        mime="image/svg+xml"
                                                    )
                                                except Exception as e:
                                                    st.error(f"Error downloading SVG: {str(e)}")
                                            
                                            with col2:
                                                try:
                                                    # Fetch PNG file from S3
                                                    png_response = st.session_state.analyzer.rag.s3_data_source.s3_client.get_object(
                                                        Bucket=st.session_state.s3_config["bucket_name"],
                                                        Key=latest_files["png"]
                                                    )
                                                    png_data = png_response['Body'].read()
                                                    st.download_button(
                                                        "Download PNG",
                                                        data=png_data,
                                                        file_name="erd.png",
                                                        mime="image/png"
                                                    )
                                                except Exception as e:
                                                    st.error(f"Error downloading PNG: {str(e)}")
                                        
                                        except Exception as e:
                                            st.error(f"Error loading relationship data: {str(e)}")
                            
                            except Exception as e:
                                st.error(f"Error generating ERD visualization: {str(e)}")
                                st.error("Please check the logs for more details.")
                
                except Exception as e:
                    st.error(f"Error analyzing schema relationships: {str(e)}")
                    st.error("Please check the logs for more details.")

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.error(
                "Please try again with a different query or check the logs for more details."
            )
