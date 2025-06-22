# Cashflow DataMap Schema API Documentation

This API provides comprehensive tools for analyzing and querying database schemas using AWS Bedrock and RAG (Retrieval-Augmented Generation).

## Base URL
```
http://0.0.0.0:8000/api/v1
```

## Authentication
Currently, the API does not require authentication.

## API Endpoints

### 1. Initialization
Initialize the analyzer with S3 configuration.

```bash
POST /initialize
```

**Request Body:**
```json
{
  "config": {
    "bucket_name": "project-quack",
    "prefix": "sbca/batch4/1299438/raw/",
    "aws_region": "us-east-1",
    "cache_size": 128,
    "pattern": ".*\\.parquet$"
  }
}
```

**Response:**
```json
{
  "message": "Analyzer initialized successfully",
  "configuration": {
    "s3": {
      "bucket_name": "project-quack",
      "prefix": "sbca/batch4/1299438/raw/",
      "region": "us-east-1",
      "pattern": ".*\\.parquet$",
      "cache_size": 128
    },
    "bedrock": {
      "region": "us-east-1",
      "embeddings_model": "amazon.titan-embed-text-v2:0",
      "inference_model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    }
  }
}
```

### 2. Schema Analysis
Analyze database schema based on natural language queries.

```bash
POST /analyze
```

**Request Body:**
```json
{
  "query": "Historical Cash Flow Analysis (Monthly)",
  "context": "Forecast Cashflow",
  "format_type": "DuckDB SQL extended with PostgreSQL syntax"
}
```

**Response:**
```json
{
  "analysis": "# Analysis of Current Payments Cashflow\n\n## SQL Query to Analyze Current Payments Cashflow\n\n```sql\nWITH payment_summary AS (\n    SELECT \n        p.date,\n        CASE \n            WHEN p.type = 'SalesReceipt' THEN 'Inflow'\n            WHEN p.type = 'PurchasePayment' THEN 'Outflow'\n            ELSE p.type\n        END AS flow_type,\n        SUM(p.total_amount) AS amount\n    FROM payments p\n    WHERE p.deleted_at IS NULL\n    GROUP BY p.date, p.type\n),\nmonthly_cashflow AS (\n    SELECT \n        TO_CHAR(TO_DATE(date, 'YYYY-MM-DD'), 'YYYY-MM') AS month,\n        flow_type,\n        SUM(amount) AS total_amount\n    FROM payment_summary\n    GROUP BY TO_CHAR(TO_DATE(date, 'YYYY-MM-DD'), 'YYYY-MM'), flow_type\n),\nnet_cashflow AS (\n    SELECT \n        month,\n        SUM(CASE WHEN flow_type = 'Inflow' THEN total_amount ELSE 0 END) AS inflows,\n        SUM(CASE WHEN flow_type = 'Outflow' THEN total_amount ELSE 0 END) AS outflows,\n        SUM(CASE WHEN flow_type = 'Inflow' THEN total_amount \n                 WHEN flow_type = 'Outflow' THEN -total_amount \n                 ELSE 0 END) AS net_cashflow\n    FROM monthly_cashflow\n    GROUP BY month\n    ORDER BY month\n)\n\nSELECT \n    month AS \"Month\",\n    ROUND(inflows::numeric, 2) AS \"Total Inflows\",\n    ROUND(outflows::numeric, 2) AS \"Total Outflows\",\n    ROUND(net_cashflow::numeric, 2) AS \"Net Cashflow\"\nFROM net_cashflow\nORDER BY month DESC;\n```",
  "query": "what is my current payments cashflow?"
}
```

### 3. SQL Generation
Generate SQL queries from natural language.

```bash
POST /sql_in_context
```

**Request Body:**
```json
{
  "query": "Historical Cash Flow Analysis (Monthly)",
  "context": "Forecast Cashflow",
  "format_type": "DuckDB SQL extended with PostgreSQL syntax"
}
```

**Response:**
```json
{
  "sql_in_context": "```sql\nWITH monthly_cash_flow AS (\n    SELECT\n        DATE_TRUNC('month', TO_DATE(le.DATE, 'YYYY-MM-DD')) AS month,\n        ba.ACCOUNT_NAME,\n        SUM(CASE WHEN le.DR > 0 THEN le.DR ELSE 0 END) AS cash_inflow,\n        SUM(CASE WHEN le.CR > 0 THEN le.CR ELSE 0 END) AS cash_outflow,\n        SUM(CASE WHEN le.DR > 0 THEN le.DR ELSE -le.CR END) AS net_cash_flow\n    FROM\n        ledger_entries le\n    JOIN\n        bank_accounts ba ON le.LEDGER_ACCOUNT_ID = ba.LEDGER_ACCOUNT_ID\n    WHERE\n        ba.ACCOUNT_TYPE_ID = 1  -- Assuming 1 represents bank accounts\n        AND le.BANK_RECONCILIATION_ID IS NOT NULL\n    GROUP BY\n        DATE_TRUNC('month', TO_DATE(le.DATE, 'YYYY-MM-DD')),\n        ba.ACCOUNT_NAME\n)\n\nSELECT\n    month,\n    ACCOUNT_NAME,\n    cash_inflow,\n    cash_outflow,\n    net_cash_flow,\n    SUM(net_cash_flow) OVER (PARTITION BY ACCOUNT_NAME ORDER BY month) AS running_balance,\n    LAG(net_cash_flow) OVER (PARTITION BY ACCOUNT_NAME ORDER BY month) AS previous_month_flow,\n    CASE\n        WHEN LAG(net_cash_flow) OVER (PARTITION BY ACCOUNT_NAME ORDER BY month) = 0 THEN NULL\n        ELSE (net_cash_flow - LAG(net_cash_flow) OVER (PARTITION BY ACCOUNT_NAME ORDER BY month)) / \n             NULLIF(ABS(LAG(net_cash_flow) OVER (PARTITION BY ACCOUNT_NAME ORDER BY month)), 0) * 100\n    END AS percentage_change\nFROM\n    monthly_cash_flow\nORDER BY\n    ACCOUNT_NAME,\n    month;\n```",
  "query": "Historical Cash Flow Analysis (Monthly)",
  "validation_status": "failed",
  "validation_message": "Error during validation: string indices must be integers, not 'str'"
}
```

### 4. Enhanced SQL Generation
Generate SQL queries using enhanced document-based embeddings with better context understanding.

```bash
POST /sql_in_context_v2
```

**Request Body:**
```json
{
  "query": "Historical Cash Flow Analysis (Monthly)",
  "context": "Forecast Cashflow",
  "format_type": "DuckDB SQL extended with PostgreSQL syntax"
}
```

**Response:**
```json
{
  "sql_in_context": "```sql\nWITH monthly_cash_flow AS (\n    SELECT\n        DATE_TRUNC('month', TO_DATE(le.DATE, 'YYYY-MM-DD')) AS month,\n        ba.ACCOUNT_NAME,\n        SUM(CASE WHEN le.DR > 0 THEN le.DR ELSE 0 END) AS cash_inflow,\n        SUM(CASE WHEN le.CR > 0 THEN le.CR ELSE 0 END) AS cash_outflow,\n        SUM(CASE WHEN le.DR > 0 THEN le.DR ELSE -le.CR END) AS net_cash_flow\n    FROM\n        ledger_entries le\n    JOIN\n        bank_accounts ba ON le.LEDGER_ACCOUNT_ID = ba.LEDGER_ACCOUNT_ID\n    WHERE\n        ba.ACCOUNT_TYPE_ID = 1  -- Assuming 1 represents bank accounts\n        AND le.BANK_RECONCILIATION_ID IS NOT NULL\n    GROUP BY\n        DATE_TRUNC('month', TO_DATE(le.DATE, 'YYYY-MM-DD')),\n        ba.ACCOUNT_NAME\n)\n\nSELECT\n    month,\n    ACCOUNT_NAME,\n    cash_inflow,\n    cash_outflow,\n    net_cash_flow,\n    SUM(net_cash_flow) OVER (PARTITION BY ACCOUNT_NAME ORDER BY month) AS running_balance,\n    LAG(net_cash_flow) OVER (PARTITION BY ACCOUNT_NAME ORDER BY month) AS previous_month_flow,\n    CASE\n        WHEN LAG(net_cash_flow) OVER (PARTITION BY ACCOUNT_NAME ORDER BY month) = 0 THEN NULL\n        ELSE (net_cash_flow - LAG(net_cash_flow) OVER (PARTITION BY ACCOUNT_NAME ORDER BY month)) / \n             NULLIF(ABS(LAG(net_cash_flow) OVER (PARTITION BY ACCOUNT_NAME ORDER BY month)), 0) * 100\n    END AS percentage_change\nFROM\n    monthly_cash_flow\nORDER BY\n    ACCOUNT_NAME,\n    month;\n```",
  "query": "Historical Cash Flow Analysis (Monthly)",
  "validation_status": "failed",
  "validation_message": "Error during validation: string indices must be integers, not 'str'"
}
```

**Key Differences from Original SQL Generation:**
- Uses enhanced document-based embeddings for better context understanding
- Improved handling of complex relationships between tables
- Better preservation of metadata and context from source documents
- More accurate SQL generation for complex queries
- Enhanced validation and error handling

### 5. Similar Schema Search
Find similar schema entries based on a query.

```bash
POST /similar
```

**Request Body:**
```json
{
  "query": "payments",
  "k": 3
}
```

**Response:**
```json
{
  "similar_schema": [
    "payments",
    "transactions",
    "payment_history"
  ],
  "scores": [
    0.95,
    0.87,
    0.82
  ]
}
```

### 6. Table Schema
Get detailed information about a specific table.

```bash
POST /table
```

**Request Body:**
```json
{
  "table_name": "artefacts"
}
```

**Response:**
```json
{
  "name": "payments",
  "columns": [
    {
      "name": "payment_id",
      "type": "string",
      "nullable": false
    },
    {
      "name": "amount",
      "type": "decimal",
      "nullable": false
    },
    {
      "name": "status",
      "type": "string",
      "nullable": false
    }
  ],
  "column_count": 3,
  "row_count": 1000,
  "last_modified": "2024-03-20T10:30:00Z",
  "size_bytes": 102400
}
```

### 7. Column Information
Get detailed information about a specific column.

```bash
POST /column
```

**Request Body:**
```json
{
  "table_name": "payments",
  "column_name": "TOTAL_NET_AMOUNT"
}
```

**Response:**
```json
{
  "name": "TOTAL_NET_AMOUNT",
  "type": "float64",
  "nullable": true,
  "unique_values": 500,
  "null_count": 0,
  "sample_values": [
    "100.00",
    "250.50",
    "75.25"
  ]
}
```

### 8. Schema Summary
Get a summary of the entire database schema.

```bash
GET /schema
```

**Response:**
```json
{
  "total_tables": 5,
  "tables": [
    {
      "name": "payments",
      "column_count": 3,
      "columns": ["payment_id", "amount", "status"],
      "row_count": 1000,
      "last_modified": "2024-03-20T10:30:00Z",
      "size_bytes": 102400
    },
    {
      "name": "customers",
      "column_count": 4,
      "columns": ["customer_id", "name", "email", "created_at"],
      "row_count": 500,
      "last_modified": "2024-03-20T10:30:00Z",
      "size_bytes": 51200
    }
  ]
}
```

### 9. ERD Generation
Generate Entity Relationship Diagrams (ERD) for the database schema.

```bash
POST /generate_erd/analyze
```

**Request Body:**
```json
{
  "query": " Analyze all relationships between all tables, their primary and foreign keys too for all tables? ",
  "context": "database_schema",
  "format_type": "json"
}
```

**Response:**
```json
{
  "enriched_schema": {
    "tables": [
      {
        "name": "payments",
        "primary_key": ["ID"],
        "foreign_keys": [
          {
            "column": "BUSINESS_ID",
            "references": {
              "table": "businesses",
              "column": "ID"
            }
          }
        ],
        "relationships": [
          {
            "related_table": "payment_artefacts",
            "type": "one-to-many",
            "description": "One payment can have multiple payment artefacts"
          }
        ]
      }
    ],
    "system_wide_relationships": [
      {
        "description": "The system represents a financial management database where payments are linked to transactions through payment_artefacts"
      }
    ]
  },
  "erd_files": {
    "json": "sbca/batch4/1299438/raw/_metadata/erd/relationships_20250513_101354.json"
  },
  "query": "Analyze relationships between payments, customers, and transactions tables",
  "context": "database_schema",
  "format_type": "json"
}
```

### 10. Get Latest ERD Files
Retrieve paths to the latest generated ERD files.

```bash
GET /generate_erd/files
```

**Response:**
```json
{
  "json":"sbca/batch4/1299438/raw/_metadata/erd/relationships_20250513_103729.json",
  "svg":null,
  "png":null
}
```

### 11. Execute SQL
Execute SQL queries against the database.

```bash
POST /execute_sql
```

**Request Body:**
```json
{
  "sql": "WITH monthly_cash_inflow AS (
    SELECT 
        DATE_TRUNC('month', CAST(bp.DATE AS DATE)) AS month,
        SUM(bp.AMOUNT) AS inflow
    FROM bank_receipts bp
    GROUP BY DATE_TRUNC('month', CAST(bp.DATE AS DATE))
  ),
  monthly_cash_outflow AS (
    SELECT 
        DATE_TRUNC('month', CAST(bp.DATE AS DATE)) AS month,
        SUM(bp.AMOUNT) AS outflow
    FROM bank_payments bp
    GROUP BY DATE_TRUNC('month', CAST(bp.DATE AS DATE))
  ),
  ledger_cash_flow AS (
    SELECT 
        DATE_TRUNC('month', CAST(le.DATE AS DATE)) AS month,
        SUM(le.CR) AS credits,
        SUM(le.DR) AS debits
    FROM ledger_entries le
    JOIN ledger_accounts la ON le.LEDGER_ACCOUNT_ID = la.ID
    WHERE la.LEDGER_ACCOUNT_TYPE_ID IN (
        SELECT ID FROM ledger_account_types 
        WHERE NAME LIKE '%Cash%' OR NAME LIKE '%Bank%'
    )
    GROUP BY DATE_TRUNC('month', CAST(le.DATE AS DATE))
  )
  SELECT 
      am.month,
      COALESCE(mci.inflow, 0) AS bank_receipts,
      COALESCE(mco.outflow, 0) AS bank_payments,
      COALESCE(lcf.credits, 0) AS ledger_credits,
      COALESCE(lcf.debits, 0) AS ledger_debits,
      COALESCE(mci.inflow, 0) - COALESCE(mco.outflow, 0) AS net_bank_flow,
      COALESCE(lcf.credits, 0) - COALESCE(lcf.debits, 0) AS net_ledger_flow
  FROM all_months am
  LEFT JOIN monthly_cash_inflow mci ON am.month = mci.month
  LEFT JOIN monthly_cash_outflow mco ON am.month = mco.month
  LEFT JOIN ledger_cash_flow lcf ON am.month = lcf.month
  ORDER BY am.month;"
}
```

**Response:**
```json
{
  "results": [
    {
      "month": "2024-03",
      "total_receipts": 150000.00,
      "total_spends": 75000.00,
      "net_cashflow": 75000.00
    },
    {
      "month": "2024-02",
      "total_receipts": 120000.00,
      "total_spends": 60000.00,
      "net_cashflow": 60000.00
    },
    {
      "month": "2024-01",
      "total_receipts": 100000.00,
      "total_spends": 50000.00,
      "net_cashflow": 50000.00
    }
  ]
}
```

## Error Handling
All endpoints return standard error responses in the following format:

```json
{
  "detail": "Error message describing what went wrong",
  "status_code": 500,
  "error_type": "ExecutionError"
}
```

## Notes
- The API uses DuckDB for SQL execution
- ERD files (SVG, PNG, JSON) are stored in S3
- All timestamps are in ISO 8601 format
- The API supports both PostgreSQL and SQLite syntax extensions
- Complex SQL queries can be executed for detailed financial analysis
- The API provides natural language to SQL conversion capabilities
- Schema analysis includes relationship mapping and ERD generation