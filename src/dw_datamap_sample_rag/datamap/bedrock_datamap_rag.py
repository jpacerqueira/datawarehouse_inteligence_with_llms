import boto3
import json
from typing import List, Dict, Any, Tuple, Optional
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import logging
from s3_data_source import S3DataSource

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BedrockDatamapRAG:
    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        region_name: str = "us-east-1",
        aws_credentials: Optional[Dict[str, str]] = None,
        cache_size: int = 128
    ):
        """Initialize the BedrockDatamapRAG class with AWS Bedrock configuration.
        
        Args:
            bucket_name (str): S3 bucket name containing parquet files
            prefix (str): Prefix for parquet files in the bucket
            region_name (str): AWS region name
            aws_credentials (Optional[Dict[str, str]]): Dictionary containing AWS credentials
            cache_size (int): Maximum number of files to cache in memory
        """
        self.region_name = region_name
        self.s3_data_source = S3DataSource(bucket_name, prefix, cache_size)
        
        try:
            # Configure AWS credentials
            if aws_credentials:
                # Use provided credentials
                aws_access_key_id = aws_credentials.get('AWS_ACCESS_KEY_ID')
                aws_secret_access_key = aws_credentials.get('AWS_SECRET_ACCESS_KEY')
                aws_session_token = aws_credentials.get('AWS_SESSION_TOKEN')
                logger.info("Using provided AWS credentials")
            else:
                # Try to get credentials from environment variables
                aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
                aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
                aws_session_token = os.getenv('AWS_SESSION_TOKEN')
                logger.info("Using AWS credentials from environment variables")
            
            if not all([aws_access_key_id, aws_secret_access_key]):
                raise ValueError("AWS credentials not found. Please provide credentials through environment variables or the aws_credentials parameter.")
            
            # Initialize boto3 client with credentials
            self.bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token
            )
            
            # Initialize Bedrock embeddings
            self.embeddings = BedrockEmbeddings(
                client=self.bedrock_client,
                model_id="amazon.titan-embed-text-v2:0"
            )
            
            self.llm = ChatBedrock(
                client=self.bedrock_client,
                model_id=inference_model,
                region_name=region_name,
                model_kwargs={
                        # Strict output control
                        "max_tokens": 10000,
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
            raise ValueError(f"Failed to initialize AWS Bedrock: {str(e)}")
    
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
                for col in table['columns']:
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
            
            self.schema_cache = {item['table_name']: item for item in schemas}
            
            # Convert schema to text
            schema_text = self._prepare_schema_text(schemas)
            if not schema_text.strip():
                raise ValueError("Generated schema text is empty")
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=30000,
                chunk_overlap=1500
            )
            texts = text_splitter.split_text(schema_text)
            
            if not texts:
                raise ValueError("No text chunks generated from schema")
            
            # Create FAISS index
            try:
                self.vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings
                )
                logger.info(f"Successfully built RAG index with {len(texts)} text chunks")
            except Exception as e:
                logger.error(f"Error creating FAISS index: {str(e)}")
                raise ValueError(f"Failed to create vector store: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error building RAG index: {str(e)}")
            raise ValueError(f"Error building RAG index: {str(e)}")
    
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
    
    def get_detailed_schema_analysis(self, query: str = "select all from payments table", context: str = "cashflow", format_type: str = "sql") -> str:
        """Get a detailed analysis of the schema based on the query."""
        if not self.vector_store:
            raise ValueError("RAG index not built. Call build_rag_index first.")
        
        try:
            # load in stack variables
            query = query
            context = context
            format_type = format_type
            # create prompt template
             prompt_template = (
                """
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
                + """ ,but, with condition to avoid UNION ALL OPERATOR and USING JOIN OPERATORS instead of UNION ALL OPERATOR.
            
            In the end of the 6 steps, show for all the detialed answered analysis:
            """
            )
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=[ "context"]
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "document_variable_name": "context",
                
                }
            )
            
            # Get response using invoke with proper input format
            response = qa_chain.invoke({
                "query": query,
            })["result"]
            logger.info(f"Response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error getting schema analysis: {str(e)}")
            raise ValueError(f"Error getting schema analysis: {str(e)}")

    def get_detailed_sql_in_context(self, query: str = "select all from payments table", context: str = "cashflow and tables payments, payment_artifacts", format_type: str = "sql") -> str:
        """Get a detailed analysis of the schema based on the query."""
        if not self.vector_store:
            raise ValueError("RAG index not built. Call build_rag_index first.")
        
        try:
            # load in stack variables
            query = query
            context = context
            format_type = format_type
            # create prompt template
            prompt_template = (
                """
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
                + """ ,but, with condition to avoid UNION ALL OPERATOR and USING JOIN OPERATORS instead of UNION ALL OPERATOR.
            
            In the end of the 6 steps, answer only in SQL format, following the notation of sql """+format_type+""" , remove extra text and comments:
            """
            )
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=[ "context"]
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "document_variable_name": "context",
                
                }
            )
            
            # Get response using invoke with proper input format
            response = qa_chain.invoke({
                "query": query,
            })["result"]
            logger.info(f"Response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error getting schema analysis: {str(e)}")
            raise ValueError(f"Error getting schema analysis: {str(e)}")
 

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get detailed schema information for a specific table."""
        try:
            response = self.rag.get_table_schema(table_name)
            
            # Convert NumPy booleans to Python booleans in columns
            if 'columns' in response:
                for column in response['columns']:
                    if 'nullable' in column:
                        column['nullable'] = bool(column['nullable'])
            
            # Rename table_name to name to match the response model
            if 'table_name' in response:
                response['name'] = response.pop('table_name')
                
            return response
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
                if os.path.basename(key).replace('.parquet', '') == table_name:
                    table_key = key
                    break
            
            if not table_key:
                raise ValueError(f"Could not find parquet file for table {table_name}")
            
            return self.s3_data_source.get_column_info(table_key, column_name)
            
        except Exception as e:
            logger.error(f"Error getting column info: {str(e)}")
            raise ValueError(f"Error getting column info: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear the S3DataSource cache."""
        self.s3_data_source.clear_cache() 