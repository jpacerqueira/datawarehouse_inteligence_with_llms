# Cash Flow Forecasting POC

A simple proof-of-concept implementation for a cash flow forecasting system. This POC demonstrates a minimal vertical slice of the system that simulates the end-to-end flow: User Input → Planning Agent → SQL Generation → Query Execution → Response.

## Overview

This POC implements a basic cash flow analysis tool that can:
- Process natural language queries about cash flow
- Translate queries into SQL using templates
- Execute SQL queries against sample data
- Display and save results on screen


## Project Structure

```
cash_flow_poc/
├── main.py                # Main application entry point
├── planning_agent.py      # Query analysis and task planning
├── schema_map.json        # Database schema definition
├── requirements.txt       # Project dependencies
├── templates/             # SQL query templates
│   ├── net_cash_flow.sql.j2
│   ├── inflow.sql.j2
│   └── outflow.sql.j2
├── data/                  # Sample data storage
│   └── payments.parquet   # Generated sample data
└── output/                # Query results storage
    └── results.json       # Example output file
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Setting up sample data

Before running queries, you need to set up the sample data:

```bash
python main.py --setup
```

This will create a sample payments dataset in the `data` directory.

### Running queries

You can run queries using the command line:

```bash
python main.py --query "Show net cash flow for February 2025"
```

Example queries:
- "Show net cash flow for February 2025"
- "What were the inflows for February 2025?"
- "Show me the expenses for February 2025"

### Output

The application will display:
1. The generated SQL query
2. The query results
3. The path to the saved results file

Results are saved in the `output` directory as JSON files with timestamps.

## Components

### User Input Interface
Simple command-line interface for entering natural language queries.

### Planning Agent
Analyzes user queries to determine intent and extract relevant parameters.

### Schema Definition
JSON file defining the database schema for reference.

### SQL Generator
Uses Jinja2 templates to generate SQL queries based on the planning agent's output.

### Execution Engine
Uses DuckDB to execute SQL queries against local Parquet files.

### Output Display
Displays results in the console and saves them to JSON files.

## Sample Flow Example

Query: "Show net cash flow for February 2025"

1. Planning Agent identifies intent as "forecast" and extracts date range
2. SQL template is rendered with appropriate parameters
3. SQL is executed against the sample data
4. Results are displayed and saved

## Future Enhancements

- Web-based UI using Streamlit
- More sophisticated query understanding
- Additional query templates
- Integration with real data sources
- Visualization of results

## Dependencies

- DuckDB: Embedded SQL database
- Pandas: Data manipulation
- Jinja2: Template rendering
- PyArrow: Parquet file support
