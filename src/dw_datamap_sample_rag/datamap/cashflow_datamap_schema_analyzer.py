from typing import List, Dict, Any, Tuple, Optional
from bedrock_datamap_rag import BedrockDatamapRAG
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataMapSchemaAnalyzer:
    def __init__(self, bucket_name: str, prefix: str = "", region_name: str = "us-east-1", cache_size: int = 128):
        """Initialize the DataMapSchemaAnalyzer with S3 configuration.
        
        Args:
            bucket_name (str): S3 bucket name containing parquet files
            prefix (str): Prefix for parquet files in the bucket
            region_name (str): AWS region name
            cache_size (int): Maximum number of files to cache in memory
        """
        # Get AWS credentials from environment variables
        aws_credentials = {
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'AWS_SESSION_TOKEN': os.getenv('AWS_SESSION_TOKEN')
        }
        
        self.rag = BedrockDatamapRAG(
            bucket_name=bucket_name,
            prefix=prefix,
            region_name=region_name,
            aws_credentials=aws_credentials,
            cache_size=cache_size
        )
        self._initialize_rag()
    
    def _initialize_rag(self, pattern: Optional[str] = None):
        """Initialize the RAG system with the S3 schema."""
        self.rag.build_rag_index(pattern)
    
    def analyze_schema(self, query: str, context: str = "cashflow", format_type: str = "sql") -> Dict[str, Any]:
        """Analyze the database schema based on a natural language query."""
        analysis = self.rag.get_detailed_schema_analysis(query, context, format_type)
        st.write(f"Analysis: {analysis}")
        return {
            "analysis": analysis,
            "query": query
        }
    
    def get_detailed_sql_in_context(self, query: str, context: str = "cashflow and tables payments, payment_artifacts", format_type: str = "sql") -> Dict[str, Any]:
        """Get detailed SQL in context."""
        return self.rag.get_detailed_sql_in_context(query, context, format_type)
    
    def get_similar_schema(self, query: str, k: int = 3) -> Tuple[List[str], List[float]]:
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
        summary = {
            "total_tables": len(all_tables),
            "tables": []
        }
        
        for table_name, schema in all_tables.items():
            table_info = {
                "name": table_name,
                "column_count": len(schema["columns"]),
                "columns": [col["name"] for col in schema["columns"]],
                "row_count": schema["row_count"],
                "last_modified": schema["last_modified"],
                "size_bytes": schema["size_bytes"]
            }
            summary["tables"].append(table_info)
        
        return summary

def create_streamlit_app():
    """Create a Streamlit app for interacting with the schema analyzer."""
    st.title("Cashflow DataMap Schema Analyzer")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    # Check if AWS credentials are available
    if not all([
        os.getenv('AWS_ACCESS_KEY_ID'),
        os.getenv('AWS_SECRET_ACCESS_KEY')
    ]):
        st.error("AWS credentials not found in environment variables. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        return
    
    # S3 configuration
    col1, col2 = st.columns(2)
    with col1:
        bucket_name = st.text_input("S3 Bucket Name:", "project-dw_intel")
    with col2:
        prefix = st.text_input("S3 Prefix:", "sbca/batch3/1299438/bronze/")
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            region_name = st.text_input("AWS Region:", "us-east-1")
        with col2:
            cache_size = st.number_input("Cache Size:", min_value=1, max_value=1000, value=128)
        pattern = st.text_input("File Pattern (optional):", ".*\\.parquet$")
    
    if st.button("Initialize Analyzer"):
        try:
            with st.spinner("Initializing schema analyzer..."):
                st.session_state.analyzer = DataMapSchemaAnalyzer(
                    bucket_name=bucket_name,
                    prefix=prefix,
                    region_name=region_name,
                    cache_size=cache_size
                )
                st.session_state.analyzer._initialize_rag(pattern)
                st.success("Analyzer initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing analyzer: {str(e)}")
    
    if st.session_state.analyzer:
        # Query input
        query = st.text_input("Enter your query about the database schema:")
        
        if query and query.strip():  # Check if query is not empty or just whitespace
            try:
                # log the query contect
                context = "cashflow"
                format_type = "PostgreSQL"
                st.write(f"Query: {query}")
                st.write(f"Context: {context}")
                st.write(f"Format Type: {format_type}")
                # Analysis tab
                with st.expander("Schema Analysis"):
                    analysis = st.session_state.analyzer.analyze_schema(query, context, format_type)
                    st.write("Analysis:")
                    st.write(analysis["analysis"])
                
                # SQL in context tab
                with st.expander("SQL in Context"):
                    sql_in_context = st.session_state.analyzer.get_detailed_sql_in_context(query, context, format_type)
                    st.write("SQL in Context:")
                    st.write(sql_in_context)
                
                # Similar schema tab
                with st.expander("Similar Schema"):
                    similar_schema, scores = st.session_state.analyzer.get_similar_schema(query)
                    for i, (text, score) in enumerate(zip(similar_schema, scores)):
                        st.write(f"Result {i+1} (Score: {score:.4f}):")
                        st.write(text)
                
                # Schema summary tab
                with st.expander("Schema Summary"):
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
                            table_schema = st.session_state.analyzer.get_table_schema(table["name"])
                            if table_schema:
                                st.write("Column Details:")
                                df = pd.DataFrame(table_schema["columns"])
                                st.dataframe(df)
                                
                                # Column analysis
                                selected_column = st.selectbox(
                                    "Select a column for detailed analysis:",
                                    [col["name"] for col in table_schema["columns"]],
                                    key=f"col_select_{table['name']}"
                                )
                                
                                if selected_column:
                                    column_info = st.session_state.analyzer.get_column_info(
                                        table["name"],
                                        selected_column
                                    )
                                    st.write("Column Analysis:")
                                    st.write(f"Type: {column_info['type']}")
                                    st.write(f"Nullable: {column_info['nullable']}")
                                    st.write(f"Unique Values: {column_info['unique_values']}")
                                    st.write(f"Null Count: {column_info['null_count']}")
                                    st.write("Sample Values:")
                                    st.write(column_info['sample_values'])
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.error("Please try again with a different query or check the logs for more details.")

if __name__ == "__main__":
    create_streamlit_app() 