import streamlit as st
import time
import numpy as np
import re
import os

import duckdb
import boto3

import logging
logger = logging.getLogger(__name__)

# Initialize global configuration variables
if "s3_config" not in st.session_state:
    st.session_state.s3_config = {
        "bucket_name": "project-dw_intel",
        "prefix": "sbca/batch3/1299438/bronze/",
        "region_name": "us-east-1",
        "cache_size": 128,
        "pattern": ".*\\.parquet$",
        "selected_files": "sbca/batch3/1299438/bronze/artefact_statuses.parquet",
    }

if "queries" not in st.session_state:
    st.session_state["queries"] = []

# Initialize DuckDB connection
if "duckdb_conn" not in st.session_state:
    st.session_state.duckdb_conn = duckdb.connect(
        ":memory:",
    )

# Initialize AWS profile
local_aws_profile = os.getenv('AWS_PROFILE', 'dw_intel-copilot-dev')

# Load AWS credentials
if "aws_credentials" not in st.session_state:
    session = boto3.Session()
    credentials = session.get_credentials()

    # Create a new connection
    conn = duckdb.connect()
    
    # install and enable the httpfs extension
    # Prerequisites: - https://duckdb.org/docs/stable/guides/network_cloud_storage/s3_import.html
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")

    # Issues workarround
    # 1. DuckDB SSO - https://github.com/duckdb/duckdb-aws/issues/14
    # 2. DuckDB S3 - https://github.com/duckdb/duckdb/issues/10409
    
    os.environ['AWS_PROFILE'] = os.getenv('AWS_PROFILE', 'dw_intel-copilot-dev')
    os.environ['AWS_REGION'] = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    os.environ['AWS_ACCESS_KEY_ID'] = credentials.access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = credentials.secret_key
    os.environ['AWS_SESSION_TOKEN'] = credentials.token

    credentials_result = conn.execute("call load_aws_credentials();")
    logger.info(f"credentials_result: {credentials_result}")
    st.session_state.aws_credentials = True

boto3.session.Session(
    profile_name=local_aws_profile,
    region_name=st.session_state.s3_config["region_name"],
)


# Function to list files in S3 bucket
def list_s3_files(bucket_name, prefix, pattern):
    st.write(
        f"Listing files in bucket: {bucket_name}, prefix: {prefix}, pattern: {pattern}"
    )
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    listed_files = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                listed_files.append(key)

    return listed_files


def load_duckdb_tables(
    bucket: str,
    prefix: str,
    tables: list[str],
):
    st.session_state.duckdb_conn.execute(
        f"""
        CREATE OR REPLACE SECRET aws_sso (
            TYPE s3,
            PROVIDER credential_chain,
            CHAIN 'sso;env',
            REGION '{st.session_state.s3_config["region_name"]}',
            PROFILE '{os.getenv('AWS_PROFILE', 'dw_intel-copilot-dev')}',
            ENDPOINT 's3.{st.session_state.s3_config["region_name"]}.amazonaws.com'
        )"""
    )

    for table in tables:
        st.write(
            f"""
                 CREATE OR REPLACE TABLE {table} AS
                 SELECT * FROM read_parquet('s3://{bucket}/{prefix}{table}.parquet')
                 """
        )
        st.session_state.duckdb_conn.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT * FROM read_parquet('s3://{bucket}/{prefix}{table}.parquet')
            """
        )


def load_sample_data(selected_table: str):
    query = f"""
    SELECT * FROM {selected_table}
    LIMIT 10
    """

    st.write(query)
    # Execute a query withing previous context
    st.session_state["queries"].append(
        (
            query,
            st.session_state.duckdb_conn.execute(query=query).df(),
        )
    )


def execute_query(query: str):
    if len(st.session_state["queries"]) and query == st.session_state["queries"][-1][0]:
        return

    st.session_state["queries"].append(
        (
            query,
            st.session_state.duckdb_conn.execute(query).df(),
        )
    )


st.markdown("# Data Analyser")

st.sidebar.header("Plotting Demo")

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

if (
    st.session_state.s3_config["bucket_name"]
    and st.session_state.s3_config["prefix"]
    and st.session_state.s3_config["pattern"]
):
    s3_files = list_s3_files(
        st.session_state.s3_config["bucket_name"],
        st.session_state.s3_config["prefix"],
        st.session_state.s3_config["pattern"],
    )

    tables = [
        re.match(
            f'{st.session_state.s3_config["prefix"]}([^/]+)\\.parquet$',
            file,
        ).group(1)
        for file in s3_files
        if re.search("([^/]+)\\.parquet$", file)
    ]

    if "tables_loaded" not in st.session_state:
        load_duckdb_tables(
            st.session_state.s3_config["bucket_name"],
            st.session_state.s3_config["prefix"],
            tables,
        )
        st.session_state.tables_loaded = True

    selected_file = st.selectbox(
        "Selected File (optional):",
        options=tables,
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

if selected_file:
    st.session_state.s3_config["table"] = selected_file

    # Load sample data
    load_sample_data(st.session_state.s3_config["table"])

new_query = st.text_area(
    "DuckDB Query",
    placeholder="SELECT * FROM artefact_statuses LIMIT 10",
    height=100,
)

if new_query:
    execute_query(new_query)

col1, col2 = st.columns(2)

with col1:

    # Function to download a file from S3
    if st.button(
        "List Files",
    ):
        files = list_s3_files(
            st.session_state.s3_config["bucket_name"],
            st.session_state.s3_config["prefix"],
            st.session_state.s3_config["pattern"],
        )

        print(files)

        st.write("## Files in S3 Bucket")
        for file in files:
            st.write(file)

if "queries" in st.session_state:
    for query, result in st.session_state["queries"]:
        st.write("## Query Result")
        st.write(f"### Query: {query}")
        st.dataframe(result)
