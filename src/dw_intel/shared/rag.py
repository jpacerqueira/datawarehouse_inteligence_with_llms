import boto3
import json
from typing import List, Dict, Any, Tuple, Optional, Generator
import botocore
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
import botocore.errorfactory
import botocore.exceptions
import botocore.validate
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv
import logging
from dw_intel.shared.s3_data_source import S3DataSource
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool
    is_primary: bool = False
    is_foreign: bool = False
    references: Optional[Dict[str, str]] = None
    relationship_type: Optional[str] = None
    # Define the schema for the column information

    def __str__(self):
        return f"{self.name} ({self.type}, {'nullable' if self.nullable else 'not nullable'})"


class TableInfo(BaseModel):
    table_name: str
    last_modified: str
    size_bytes: int
    row_count: int
    columns: List[ColumnInfo]
    # Define the schema for the table information

    def __str__(self):
        return (
            f"Table Name: {self.table_name}\n"
            f"Last Modified: {self.last_modified}\n"
            f"Size (bytes): {self.size_bytes}\n"
            f"Row Count: {self.row_count}\n"
            f"Columns: {'\n'.join([str(col) for col in self.columns])}\n"
        )

class StreamingBufferCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming output to a buffer."""
    
    def __init__(self):
        self.buffer = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Add new token to buffer."""
        self.buffer.append(token)
        
    def get_buffer(self) -> str:
        """Get the complete buffer as a string."""
        return "".join(self.buffer)
        
    def clear_buffer(self) -> None:
        """Clear the buffer."""
        self.buffer = []

class BedrockDatamapRAG:
    schema_cache: Dict[str, Any] = {}
    schema_cache_v2: Dict[str, TableInfo] = {}
    schema_cache_v3: List[Document] = []
    vector_store: Optional[FAISS] = None
    vector_store_v2: Optional[FAISS] = None
    vector_store_v3: Optional[FAISS] = None
    s3_data_source: S3DataSource
    bedrock_client: BedrockRuntimeClient
    embeddings: BedrockEmbeddings
    llm: ChatBedrock

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        region_name: str = "us-east-1",
        embeddings_model: str = "amazon.titan-embed-text-v2:0",
        inference_model: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        cache_size: int = 128,
    ):
        """Initialize the BedrockDatamapRAG class with AWS Bedrock configuration.

        Args:
            bucket_name (str): S3 bucket name containing parquet files
            prefix (str): Prefix for parquet files in the bucket
            region_name (str): AWS region name
            embeddings_model (str): Model ID for Bedrock embeddings
            inference_model (str): Model ID for Bedrock inference
            cache_size (int): Maximum number of files to cache in memory
        """
        self.region_name = region_name
        self.s3_data_source = S3DataSource(bucket_name, prefix, cache_size)

        try:
            session = boto3.Session()

            # Initialize boto3 client with credentials
            self.bedrock_client = session.client(
                service_name="bedrock-runtime",
                region_name=region_name,
            )

            # Initialize Bedrock embeddings
            self.embeddings = BedrockEmbeddings(
                client=self.bedrock_client,
                model_id=embeddings_model,
                region_name=region_name
            )

            self.llm = ChatBedrock(
                client=self.bedrock_client,
                model_id=inference_model,
                region_name=region_name,
                model_kwargs={
                        # Strict output control
                        "max_tokens": 131000,        # Max Token = 131072 ==> InvokeModel operation: The maximum tokens you requested exceeds the model limit of 131072
                        "temperature": 0.0,          # Minimize randomness (0-1 scale)
                        "top_p": 0.2,                # Narrow token selection (0.2-0.3 for precision)
                        "top_k": 10,                  # Consider only top token candidates
                        # Precision-focused parameters
                        "system": "You are an expert assistant that provides accurate, detailed, and factual responses from your analysis and also are an expert in SQL. Avoid speculation and focus on verifiable information.",
                        # Validation parameters
                        "stop_sequences": ["\n\nHuman:"]  # Prevent open-ended responses
                    }
            )

            # Test the connection by making a simple embedding request
            test_embedding = self.embeddings.embed_query("test")
            if not test_embedding:
                raise ValueError("Failed to get test embedding from Bedrock")

            logger.info("Successfully initialized Bedrock client and models")

        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock: {str(e)}")
            raise ValueError(f"Failed to initialize AWS Bedrock: {str(e)}") from e

    def _prepare_schema_text(self, schemas: List[Dict[str, Any]]) -> str:
        """Convert schema information to text format for embedding."""
        try:
            text_chunks = []
            for table in schemas:
                table_text = f"Table: {table['table_name']}\n"
                table_text += f"Last Modified: {table['last_modified']}\n"
                table_text += f"Size: {table['size_bytes']} bytes\n"
                table_text += f"Row Count: {table['row_count']}\n"
                table_text += "Columns:\n"
                for col in table["columns"]:
                    table_text += f"- {col['name']} ({col['type']}, {'nullable' if col['nullable'] else 'not nullable'})\n"
                text_chunks.append(table_text)

            return "\n\n".join(text_chunks)

        except Exception as e:
            logger.error(f"Error preparing schema text: {str(e)}")
            raise ValueError(f"Error preparing schema text: {str(e)}")

    def build_rag_index(self, pattern: Optional[str] = None):
        """Build the RAG index from S3 parquet files schema.

        Args:
            pattern (Optional[str]): Regex pattern to filter filenames
        """
        try:
            # Get schema from S3
            schemas = self.s3_data_source.get_all_schemas(pattern)
            if not schemas:
                raise ValueError("No parquet files found in the specified S3 location")

            self.schema_cache = {item["table_name"]: item for item in schemas}

            # Convert schema to text
            schema_text = self._prepare_schema_text(schemas)
            if not schema_text.strip():
                raise ValueError("Generated schema text is empty")

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=30000,
                chunk_overlap=100
            )
            texts = text_splitter.split_text(schema_text)

            if not texts:
                raise ValueError("No text chunks generated from schema")

            # Create FAISS index
            try:
                self.vector_store = FAISS.from_texts(
                    texts=texts, embedding=self.embeddings
                )
                logger.info(
                    f"Successfully built RAG index with - vector_store - text embeddings - {len(texts)} text chunks"
                )
            except Exception as e:
                logger.error(f"Error creating FAISS index: {str(e)}")
                raise ValueError(f"Failed to create vector store: {str(e)}") from e

        except Exception as e:
            logger.error(f"Error building RAG index: {str(e)}")
            raise ValueError(f"Error building RAG index: {str(e)}") from e

    def _prepare_schema_text_v2(self, schemas: List[Dict[str, Any]]) -> dict[str, TableInfo]:
        """Convert schema information to text format for embedding."""

        tables = {}
        for table in schemas:
            table_info = TableInfo(
                table_name=table["table_name"],
                last_modified=table["last_modified"],
                size_bytes=table["size_bytes"],
                row_count=table["row_count"],
                columns=[
                    ColumnInfo(
                        name=col["name"],
                        type=col["type"],
                        nullable=col["nullable"],
                    )
                    for col in table["columns"]
                ],
            )
            tables[table_info.table_name] = table_info

        return tables

    def build_rag_index_v2(self, pattern: Optional[str] = None):
        """Build the RAG index v2 as documents from tables loaded in schema cache of text.

        Args:
            pattern (Optional[str]): Regex pattern to filter filenames
        """
        if not self.schema_cache or len(self.schema_cache.keys()) == 0:
            logger.info("Schema cache is empty. Loading schemas from S3.")
            self.load_schemas_from_s3(pattern)

        try:
            # Convert schema_cache to list of schemas
            schemas_list = list(self.schema_cache.values())
            
            # Parse all schemas
            self.schema_cache_v2 = self._prepare_schema_text_v2(schemas_list)

            # Embed schema information
            docs_to_embed = [
                Document(page_content=str(table_info))
                for table_info in self.schema_cache_v2.values()
            ]

            # Create FAISS index
            self.vector_store_v2 = FAISS.from_documents(
                documents=docs_to_embed,
                embedding=self.embeddings,
            )
            logger.info(
                    f"Successfully built RAG index with - vector_store_v2 - tables from_documents embeddings - {len(docs_to_embed)} documents or tables"
                )

        except Exception as e:
            logger.error(f"Error preparing schema_cache_v2 text: {str(e)}")

    def query_schema(self, query: str, k: int = 3) -> Tuple[List[str], List[float]]:
        """Query the schema using RAG and return relevant information with similarity scores."""
        if not self.vector_store:
            raise ValueError("RAG index not built. Call build_rag_index first.")

        try:
            # Get relevant documents
            docs = self.vector_store.similarity_search_with_score(query, k=k)

            # Extract text and scores
            texts = [doc[0].page_content for doc in docs]
            scores = [doc[1] for doc in docs]

            return texts, scores

        except Exception as e:
            logger.error(f"Error querying schema: {str(e)}")
            raise ValueError(f"Error querying schema: {str(e)}")

    def _create_streaming_qa_chain(self, prompt: PromptTemplate) -> RetrievalQA:
        """Create a streaming QA chain with buffer."""
        streaming_handler = StreamingBufferCallbackHandler()
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_v2.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "context",  # Only use context as document variable
            },
            callbacks=[streaming_handler]
        )

    def _create_streaming_qa_chain_v2(self, prompt: PromptTemplate) -> RetrievalQA:
        """Create a streaming QA chain with buffer."""
        streaming_handler = StreamingBufferCallbackHandler()
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_v2.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "tables",  # Only use tables as document variable
            },
            callbacks=[streaming_handler]
        )
    
    def get_detailed_sql_in_context(
        self,
        query: str,
        context: str = "cashflow",
        format_type: str = "sql",
    ) -> str:
        """Get SQL in context based on the query and schema."""
        if not self.vector_store:
            raise ValueError("RAG index not built. Call build_rag_index first.")

        try:
            # Create prompt template for SQL generation
            prompt_template = (
                """
            You are a DataWareHouse SQL pipelines expert, in SQL single transactional queries. You are given a query and a context, to generate a SQL query, in a format type.
            Based on the following database schema, context, and query, provide a detailed answer to the analysis in output format, """
                + format_type
                + """ :
            
            Query: """
                + query
                + """
            
            Context: {context}
            
            Please execute internally the following comprehensive analysis including:
            1. Table structure and relationships
            2. Data types and constraints
            3. Data volume and freshness
            4. Query optimization recommendations
            5. Potential data quality issues
            6. A query in SQL format """
                + format_type
                + """ with the following requirements:
                - Use JOIN operators instead of UNION ALL
                - Ensure proper type casting for all columns:
                  * When using string functions (REPLACE, SUBSTRING, etc.), first CAST to VARCHAR
                  * When converting to numeric types, ensure proper decimal precision
                  * Handle NULL values appropriately with COALESCE or IFNULL
                  * Use explicit CAST statements for all type conversions
                - Follow this pattern for type conversions:
                  * String to Number: CAST(CAST(string_column AS VARCHAR) AS DECIMAL(precision,scale))
                  * Number to String: CAST(numeric_column AS VARCHAR)
                  * Date conversions: CAST(date_column AS DATE) or CAST(date_column AS TIMESTAMP)
                - Only use functions compatible with the SQL format syntax requested
                - Validate all function arguments match expected types
                - Handle special characters and formatting in string operations
                - If present the column is_latets_record in table, to avoid query duplicates, use filter is_latest_record=1
            
            In the end of the 6 steps, answer only in SQL format, following the notation of sql """+format_type+""" , remove extra text and comments:
            """
            )

            prompt = PromptTemplate(
                template=prompt_template, input_variables=["context"]
            )

            # Create streaming QA chain
            qa_chain = self._create_streaming_qa_chain(prompt)

            # Get response with streaming
            response = qa_chain.invoke(
                {
                    "query": query,
                }
            )["result"]
            
            # Log the complete response
            logger.info(f"Generated SQL: {response}")
            return response

        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            raise ValueError(f"Error generating SQL: {str(e)}") from e

    def get_detailed_schema_analysis(
        self,
        query: str,
        context: str = "cashflow",
        format_type: str = "sql",
    ) -> str:
        """Get a detailed analysis of the schema based on the query."""
        if not self.vector_store:
            raise ValueError("RAG index not built. Call build_rag_index first.")

        try:
            # Create enhanced prompt template
            prompt_template = (
                """
            You are a DataWareHouse expert in Analsysis, Data Profiling and SQL pipelines expert. You are given a flavour of SQL query and a context, to generate tables analysis and a SQL query over those tables.
            Based on the following database schema, context, and query, provide a detailed answer to the analysis in output format, """
                + format_type
                + """ :
            
            Query: """
                + query
                + """
            
            Context: {context}
            
            Please execute the following comprehensive analysis including:
            1. Table structure and relationships
            2. Data types and constraints
            3. Data volume and freshness
            4. Query optimization recommendations
            5. Potential data quality issues
            6. A query in SQL format """
                + format_type
                + """ with the following requirements:
                - Use JOIN operators instead of UNION ALL
                - Ensure proper type casting for all columns:
                  * When using string functions (REPLACE, SUBSTRING, etc.), first CAST to VARCHAR
                  * When converting to numeric types, ensure proper decimal precision
                  * Handle NULL values appropriately with COALESCE or IFNULL
                  * Use explicit CAST statements for all type conversions
                - Follow this pattern for type conversions:
                  * String to Number: CAST(CAST(string_column AS VARCHAR) AS DECIMAL(precision,scale))
                  * Number to String: CAST(numeric_column AS VARCHAR)
                  * Date conversions: CAST(date_column AS DATE) or CAST(date_column AS TIMESTAMP)
                - Only use functions compatible with the SQL format syntax requested
                - Validate all function arguments match expected types
                - Handle special characters and formatting in string operations
                - If present the column is_latets_record in table, to avoid query duplicates, use filter is_latest_record=1
            
            Based on the following database schema, in the embedding vector store, analyze and identify in tables and columns relationships, primary keys and foreign keys:
            7. Primary Keys for each table
            8. Foreign Key relationships between tables
            9. Table relationships and cardinality
            10. Column dependencies and constraints
            11. Provide a detailed analysis in JSON format with the following structure:

            In the end of the 6 steps, show for all the detailed answered analysis:
            """
            )

            prompt = PromptTemplate(
                template=prompt_template, input_variables=["context"]
            )
            
            # Log the prompt and parameters
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Query: {query}")
            logger.info(f"Context: {context}")
            logger.info(f"Format Type: {format_type}")
            logger.info(f"Model: {self.llm.model_id}")

            # Create streaming QA chain
            qa_chain = self._create_streaming_qa_chain(prompt)

            # Get response with streaming
            response = qa_chain.invoke(
                {
                    "query": query,
                }
            )["result"]
            
            # Log the complete response
            logger.info(f"Analysis: {response}")
            return response

        except Exception as e:
            logger.error(f"Error getting schema analysis: {str(e)}")
            raise ValueError(f"Error getting schema analysis: {str(e)}") from e

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get detailed schema for a specific table."""
        try:
            response = self.schema_cache.get(table_name, {}) # TODO: input == add output
            output = response
            # log output
            logger.info(f"Table schema: {output}")
            return output
        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            raise ValueError(f"Error getting table schema: {str(e)}") from e

    def get_column_info(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific column in a table."""
        try:
            table_schema = self.get_table_schema(table_name)
            if not table_schema:
                raise ValueError(f"Table {table_name} not found")

            # Find the file key for the table
            table_key = None
            for key in self.s3_data_source.list_parquet_files():
                if os.path.basename(key).replace(".parquet", "") == table_name:
                    table_key = key
                    break

            if not table_key:
                raise ValueError(f"Could not find parquet file for table {table_name}")

            return self.s3_data_source.get_column_info(table_key, column_name)

        except Exception as e:
            logger.error(f"Error getting column info: {str(e)}")
            raise ValueError(f"Error getting column info: {str(e)}") from e

    def clear_cache(self) -> None:
        """Clear the S3DataSource cache."""
        self.s3_data_source.clear_cache()

    def get_detailed_erd_schema(
        self,
        query: str = "Analyze all tables and their relationships",
        context: str = "database_schema",
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """Get a detailed analysis of table relationships and schema structure using LLM.
        
        Args:
            query (str): Query to analyze relationships
            context (str): Context for the analysis
            format_type (str): Output format type
            
        Returns:
            Dict[str, Any]: Enriched schema information with relationships
        """
        if not self.vector_store_v2:
            raise ValueError("RAG index v2 not built. Call build_rag_index_v2 first.")

        try:
            # table format # created conflict with PromptTemplate - escaped json notation
            table_format = """ \\{ "tables": \\{ "table_name": \\{ "primary_keys": ["column1", "column2"], "foreign_keys": [ \\{ "column": "column_name", "references": \\{ "table": "referenced_table", "column": "referenced_column" \\}, "relationship_type": "one-to-one|one-to-many|many-to-many" \\} ], "columns": [ \\{ "name": "column_name", "type": "data_type", "nullable": true, "is_primary": false, "is_foreign": false, "description": "column description" \\} ]\\} \\},"relationships": [\\{ "from_table": "table1", "to_table": "table2", "type": "one-to-one|one-to-many|many-to-many", "columns": [ \\{ "from": "column1", "to": "column2"\\} \\] \\} \\] \\} """
            
            # Create enhanced prompt template for ERD analysis
            prompt_template = """
            You are a DataWareHouse Entity Relationship Diagram ERD expert. You are given a query, a format type and a context, to generate an ERD diagram in an ouput format.
            Based on the following database schema, in the embedding vector store, analyze and identify in tables and columns relationships, primary keys and foreign keys:
            1. Primary Keys for each table
            2. Foreign Key relationships between tables
            3. Table relationships and cardinality
            4. Column dependencies and constraints
            5. Provide a detailed analysis in JSON format with the following structure:

            Augmented knowledge from the context and the query, for the reduced output format:
            Query: """+query+"""
            
            Database Schema Context: """+context+"""

            Output Format: """+format_type+"""
            
            Please execute a detailed analysis, and provide an output in """+format_type+""" format, with no additional text, but with
            all information enriched from the context, the query and the embedding vector store, 
            in a """+format_type+""" format defined as:
            {tables}
            
            Focus analysis on:
            1. Identifying natural primary keys and foreign keys, from data types and column names in the embedding vector store
            2. Finding foreign key relationships based on naming patterns and data types in the embedding vector store
            3. Inferring relationship types based on column uniqueness in the embedding vector store
            4. Documenting column dependencies in the embedding vector store
            5. Providing clear relationship descriptions in the embedding vector store
            
            Return only the JSON structure, above, no additional text.
            """

            tables = table_format

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["tables"]
            )

            # Log the prompt and parameters
            logger.info("--------------------------------")
            logger.info("ERD Schema Prompt")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Query: {query}")
            logger.info(f"Context: {context}")
            logger.info(f"Format Type: {format_type}")
            logger.info(f"Model: {self.llm.model_id}")
            logger.info("--------------------------------")

            # Create streaming QA chain
            qa_chain = self._create_streaming_qa_chain_v2(prompt)

            # Get response with streaming
            response = qa_chain.invoke(
                {
                    "query": query,
                }
            )["result"]

            # Log the response
            logger.info("--------------------------------")
            logger.info("ERD Schema Response")
            logger.info(f"Get ERD schema - speculative Response: {response}")
            logger.info("--------------------------------")

            # Initialize the enriched schema
            enriched_schema = {"tables": {}, "relationships": []}

            # trim response to JSON only
            try:
                if "```json" in response:
                    response = response.split("```json", 1)[1].strip().replace("```", "").strip()
                elif "```" in response:
                    # Handle case where there's a code block but no language specified
                    response = response.split("```", 1)[1].strip().replace("```", "").strip()
                
                # Clean up any remaining markdown or extra text
                response = response.strip()
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()
                
                logger.info(f"Get ERD schema - speculative Response JSON trimmed: {response}")
                
                # Parse the response
                enriched_schema = json.loads(response)
                logger.info(f"Get ERD schema - speculative Enriched schema: {enriched_schema}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                logger.error(f"Raw response: {response}")
                raise ValueError(f"Invalid JSON response from LLM: {str(e)}")

            # log schema_cache_v2
            logger.info(f"Get ERD schema - speculative Schema cache v2: {self.schema_cache_v2}")
            
            # Update schema_cache_v2 with enriched information
            for table_info in enriched_schema['tables']:
                # log table_name
                table_name = table_info['name']
                logger.info(f"Get ERD schema - speculative Table name: {table_name}")
                
                if table_name in self.schema_cache_v2:
                    # Update existing table info with relationship data
                    current_table = self.schema_cache_v2[table_name]
                    # log current_table
                    logger.info(f"Get ERD schema - speculative Current table: {current_table}")
                    
                    # Create a new list of columns with updated metadata
                    updated_columns = []
                    for col in current_table.columns:
                        # Create a new ColumnInfo with the same base attributes
                        updated_col = ColumnInfo(
                            name=col.name,
                            type=col.type,
                            nullable=col.nullable,
                            is_primary=col.name in table_info.get('primary_key', []),
                            is_foreign=False,
                            references=None,
                            relationship_type=None
                        )
                        
                        # Check if column is a foreign key
                        for fk in table_info.get('foreign_keys', []):
                            if col.name == fk['column']:
                                updated_col.is_foreign = True
                                updated_col.references = fk['references']
                                updated_col.relationship_type = fk.get('relationship_type', 'unknown')
                        
                        updated_columns.append(updated_col)
                    
                    # Update the table's columns with the new list
                    current_table.columns = updated_columns
            
            logger.info(f"Successfully enriched schema with relationship information for {len(enriched_schema['tables'])} tables")
            return enriched_schema

        except Exception as e:
            logger.error(f"Error getting detailed ERD schema: {str(e)}")
            raise ValueError(f"Error getting detailed ERD schema: {str(e)}") from e

    def _load_hydration_data(self) -> str:
        """Load and process hydration data files for embeddings.
        
        Returns:
            str: Combined text content from all hydration data files
        """
        try:
            # TODO: add use_cases_schema.txt to the hydration data files - for use_cases prefix in s3_data_source
            if "/use_cases/" in self.s3_data_source.prefix:
                hydration_files = [
                    "1.prompt.md.txt",
                    "2.entity-relationship-diagram.md.txt",
                    "3.SQLGeneration.md.txt",
                    "1-1.json-parsed-schema.json.txt",
                    "use_cases_schema.txt"
                ]
            else:
                hydration_files = [
                "1.prompt.md.txt",
                "2.entity-relationship-diagram.md.txt",
                "3.SQLGeneration.md.txt",
                "1-1.json-parsed-schema.json.txt"
                ]
            
            combined_text = []
            for file_name in hydration_files:
                file_path = os.path.join("/app/hydration_data", file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Add file content with a separator
                        combined_text.append(f"=== Content from {file_name} ===\n{content}\n")
                        logger.info(f"Loaded hydration data from {file_name}")
                else:
                    logger.warning(f"Hydration data file not found: {file_path}")
            
            if not combined_text:
                raise ValueError("No hydration data files were loaded")
                
            return "\n".join(combined_text)
            
        except Exception as e:
            logger.error(f"Error loading hydration data: {str(e)}")
            raise ValueError(f"Error loading hydration data: {str(e)}") from e

    def _create_documents_from_text_chunks(self, text_chunks: List[str]) -> List[Document]:
        """Convert text chunks to documents with metadata.
        
        Args:
            text_chunks (List[str]): List of text chunks to convert
            
        Returns:
            List[Document]: List of documents with metadata
        """
        documents = []
        for i, chunk in enumerate(text_chunks):
            # Extract file name from the chunk if it exists
            file_name = "unknown"
            if "=== Content from" in chunk:
                file_name = chunk.split("=== Content from")[1].split("===")[0].strip()
            
            # Split chunk into smaller parts if it's too large
            # Using a conservative limit of 40000 to account for metadata and overhead
            max_chunk_size = 5000
            if len(chunk) > max_chunk_size:
                # Split into smaller chunks while preserving file markers
                sub_chunks = []
                current_chunk = ""
                lines = chunk.split('\n')
                
                for line in lines:
                    if len(current_chunk) + len(line) + 1 > max_chunk_size:
                        if current_chunk:
                            sub_chunks.append(current_chunk)
                        current_chunk = line
                    else:
                        if current_chunk:
                            current_chunk += '\n'
                        current_chunk += line
                
                if current_chunk:
                    sub_chunks.append(current_chunk)
                
                # Create documents for each sub-chunk
                for j, sub_chunk in enumerate(sub_chunks):
                    doc = Document(
                        page_content=sub_chunk,
                        metadata={
                            "source": file_name,
                            "chunk_index": f"{i}.{j}",
                            "type": "hydration_data",
                            "version": "v3",
                            "is_subchunk": True,
                            "parent_chunk": i
                        }
                    )
                    documents.append(doc)
            else:
                # Create document for the original chunk
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_name,
                        "chunk_index": str(i),
                        "type": "hydration_data",
                        "version": "v3",
                        "is_subchunk": False
                    }
                )
                documents.append(doc)
        
        return documents

    def build_rag_index_v3(self, pattern: Optional[str] = None):
        """Build the RAG index v3 using document-based embeddings from hydration data.
        
        This version uses the hydration data files from /app/hydration_data
        to create document-based embeddings with proper pagination.

        Args:
            pattern (Optional[str]): Regex pattern to filter filenames (not used in v3)
        """
        try:
            # Get the text chunks for vector_store_v3
            hydration_text = self._load_hydration_data()
            if not hydration_text:
                raise ValueError("No hydration data files found")

            # Split text into chunks using smaller size for initial split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=20000,  # Reduced from 30000 to allow for metadata
                chunk_overlap=100  # Reduced overlap to prevent token limit issues
            )
            text_chunks = text_splitter.split_text(hydration_text)

            if not text_chunks:
                raise ValueError("No text chunks generated from hydration data")

            # Convert text chunks to documents with pagination
            documents = self._create_documents_from_text_chunks(text_chunks)

            # Parse all schemas for v3 vector store cache
            self.schema_cache_v3 = documents

            # log schema_cache_v3
            logger.info(f"Schema cache v3: {len(self.schema_cache_v3)} documents created")

            # Create FAISS index from documents
            try:
                self.vector_store_v3 = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                logger.info(
                    f"Successfully built RAG index v3 with document embeddings - {len(documents)} documents"
                )
            except Exception as e:
                logger.error(f"Error creating FAISS index v3: {str(e)}")
                raise ValueError(f"Failed to create vector store v3: {str(e)}") from e

        except Exception as e:
            logger.error(f"Error building RAG index v3: {str(e)}")
            raise ValueError(f"Error building RAG index v3: {str(e)}") from e

    def _create_streaming_qa_chain_v3(self, prompt: PromptTemplate) -> RetrievalQA:
        """Create a streaming QA chain with buffer for v4."""
        streaming_handler = StreamingBufferCallbackHandler()
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_v3.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "context",
            },
            callbacks=[streaming_handler]
        )

    def get_detailed_sql_in_context_v2(
        self,
        query: str,
        context: str = "cashflow",
        format_type: str = "sql",
    ) -> str:
        """Get SQL in context based on the query and schema using v4 vector store.
        
        This version uses the v3 vector store which is built from document-based
        embeddings derived from /app/hydration_data files content, providing enhanced context understanding
        and metadata awareness for SQL generation.
        
        Args:
            query (str): The SQL query to analyze
            context (str): The context for the analysis
            format_type (str): The output format type
            
        Returns:
            str: The generated SQL query
        """
        if not self.vector_store_v3:
            raise ValueError("RAG index v3 not built. Call build_rag_index_v3 first.")

        try:
            # Create prompt template for SQL generation
            prompt_template = (
                """
            You are a DataWareHouse SQL pipelines expert, in SQL single transactional queries. You are given a query and a context, to generate a SQL query, in a format type.
            Based on the following database schema, context, and query, provide a detailed answer to the analysis in output format, """
                + format_type
                + """ :
            
            Query: """
                + query
                + """
            
            Context: {context}
            
            Please execute internally the following comprehensive analysis including:
            1. Table structure and relationships
            2. Data types and constraints
            3. Data volume and freshness
            4. Query optimization recommendations
            5. Potential data quality issues
            6. A query in SQL format """
                + format_type
                + """ with the following requirements:
                - Use JOIN operators instead of UNION ALL
                - Ensure proper type casting for all columns:
                  * When using string functions (REPLACE, SUBSTRING, etc.), first CAST to VARCHAR
                  * When converting to numeric types, ensure proper decimal precision
                  * Handle NULL values appropriately with COALESCE or IFNULL
                  * Use explicit CAST statements for all type conversions
                - Follow this pattern for type conversions:
                  * String to Number: CAST(CAST(string_column AS VARCHAR) AS DECIMAL(precision,scale))
                  * Number to String: CAST(numeric_column AS VARCHAR)
                  * Date conversions: CAST(date_column AS DATE) or CAST(date_column AS TIMESTAMP)
                - Only use functions compatible with the SQL format syntax requested
                - Validate all function arguments match expected types
                - Handle special characters and formatting in string operations
                - If present the column is_latets_record in table, to avoid query duplicates, use filter is_latest_record=1
            
            In the end of the 6 steps, answer only in SQL format, following the notation of sql """+format_type+""" , remove extra text and comments:
            """
            )

            prompt = PromptTemplate(
                template=prompt_template, input_variables=["context"]
            )

            # Create streaming QA chain
            qa_chain = self._create_streaming_qa_chain_v3(prompt)

            # Get response with streaming
            response = qa_chain.invoke(
                {
                    "query": query,
                }
            )["result"]
            
            # Log the complete response
            logger.info(f"Generated SQL (v2): {response}")
            return response

        except Exception as e:
            logger.error(f"Error generating SQL (v2): {str(e)}")
            raise ValueError(f"Error generating SQL (v2): {str(e)}") from e