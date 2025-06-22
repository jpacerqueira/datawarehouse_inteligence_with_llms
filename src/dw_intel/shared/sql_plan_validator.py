import os
import logging
import json
import re
import time
from typing import Dict, Any, List, Optional, Tuple

from fastapi import HTTPException

logger = logging.getLogger(__name__)

class SQLPlanValidator:
    """A class to validate SQL plans before execution."""
    
    def __init__(self, schema_cache: Dict[str, Any]):
        """Initialize the SQLPlanValidator with schema information.
        
        Args:
            schema_cache (Dict[str, Any]): Dictionary containing table schemas
        """
        self.schema_cache = schema_cache
        self.validated_plans = {}
        
    def validate_sql_plan(self, sql_plan: str, context: str = "cashflow") -> Tuple[bool, str]:
        """Validate a SQL plan for correctness and alignment with available tables.
        
        Args:
            sql_plan (str): The SQL plan to validate
            context (str): The context in which the SQL is being used
            
        Returns:
            Tuple[bool, str]: (is_valid, validation_message)
        """
        try:
            # Clean and normalize the SQL plan
            cleaned_sql = self.clean_sql_plan(sql_plan)
            
            # Extract and categorize table names
            table_categories = self._extract_table_names(cleaned_sql)
            # log the table categories
            logger.info(f"Table categories: {table_categories}")
            permanent_tables = table_categories['permanent_tables']
            
            # Validate only permanent tables against schema
            missing_tables = self._validate_tables_exist(permanent_tables)
            if missing_tables:
                return False, f"Tables not found in schema: {', '.join(missing_tables)}"
            
            # Validate column references for permanent tables only
            column_issues = self._validate_column_references(cleaned_sql, permanent_tables)
            if column_issues:
                return False, f"Column validation issues: {', '.join(column_issues)}"
            
            # Validate SQL syntax
            syntax_issues = self._validate_sql_syntax(cleaned_sql)
            if syntax_issues:
                return False, f"SQL syntax issues: {', '.join(syntax_issues)}"
            
            # Store validated plan with table categorization
            self.validated_plans[cleaned_sql] = {
                "tables": table_categories,
                "context": context,
                "timestamp": time.time()
            }
            
            return True, "SQL plan validated successfully"
            
        except Exception as e:
            logger.error(f"Error validating SQL plan: {str(e)}")
            return False, f"Error during validation: {str(e)}"
    
    def clean_sql_plan(self, sql_plan: str) -> str:
        """Clean and normalize the SQL plan.
        
        Args:
            sql_plan (str): The SQL plan to clean
            
        Returns:
            str: Cleaned SQL plan
        """
        # Remove LLM generated comments and analysis before SQL
        if "```sql" in sql_plan:
            sql_trimmed = sql_plan.split("```sql", 1)[1].strip()
            # Remove markdown code blocks if present
            sql = sql_trimmed.replace("```sql", "").replace("```", "").strip()
        else:
            sql = sql_plan.strip()
        
        # Remove multiple spaces and newlines
        #sql = re.sub(r'\s+', ' ', sql)
        
        # Ensure proper spacing around operators
        sql = re.sub(r'([=<>!])\s*([=<>!])', r'\1 \2', sql)
        
        return sql
    
    def _extract_table_names(self, sql: str) -> Dict[str, List[str]]:
        """Extract table names from SQL query and categorize them.
        
        Args:
            sql (str): The SQL query
            
        Returns:
            Dict[str, List[str]]: Dictionary with 'cte_tables' and 'permanent_tables' lists
        """
        # Convert to lowercase for case-insensitive matching
        # clean comment lines from sql validation if start with -- , or if -- in the middle of the line after n spaces
        sql_lower = re.sub(r'^\s*--.*$', '', sql, flags=re.MULTILINE)
        sql_lower = re.sub(r'\s*--.*$', '', sql_lower, flags=re.MULTILINE)
        sql_lower = sql_lower.lower()

        # log the sql_lower
        logger.info(f"SQL lower - extract table names: {sql_lower}")
        
        # Extract CTE names
        cte_pattern = r'(?:with\s+)?([a-zA-Z0-9_]+)\s+as\s*\('
        cte_tables = re.findall(cte_pattern, sql_lower)
        
        # Find table names after FROM and JOIN clauses
        table_pattern = r'(?:from|join)\s+([a-zA-Z0-9_]+)'
        all_tables = re.findall(table_pattern, sql_lower)
        
        # Separate permanent tables (those not in CTEs)
        permanent_tables = [table for table in all_tables if table not in cte_tables]

        # log the tables
        logger.info(f"CTE tables: {cte_tables}")
        logger.info(f"Permanent tables: {permanent_tables}")
        
        return {
            'cte_tables': list(set(cte_tables)),
            'permanent_tables': list(set(permanent_tables))
        }
    
    def _validate_tables_exist(self, table_names: List[str]) -> List[str]:
        """Validate that all referenced tables exist in the schema.
        
        Args:
            table_names (List[str]): List of table names to validate
            
        Returns:
            List[str]: List of missing table names
        """
        missing_tables = []
        # log the schema cache
        #logger.info(f"---  _validate_tables_exist - Schema cache: {self.schema_cache}")
        # log the table names
        #logger.info(f"---  _validate_tables_exist - Table names: {table_names}")
        for table in table_names:
            if table not in self.schema_cache:
                missing_tables.append(table)
        return missing_tables
    
    def _validate_column_references(self, sql: str, table_names: List[str]) -> List[str]:
        """Validate that all column references are valid for the referenced tables.
        
        Args:
            sql (str): The SQL query
            table_names (List[str]): List of table names in the query
            
        Returns:
            List[str]: List of column validation issues
        """
        issues = []
        
        # Extract column references
        column_pattern = r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)'
        column_refs = re.findall(column_pattern, sql)
        
        for table, column in column_refs:
            if table in table_names:
                table_schema = self.schema_cache.get(table, {})
                columns = [col["name"] for col in table_schema.get("columns", [])]
                if column not in columns:
                    issues.append(f"Column '{column}' not found in table '{table}'")
        
        return issues
    
    def _validate_sql_syntax(self, sql: str) -> List[str]:
        """Validate basic SQL syntax rules.
        
        Args:
            sql (str): The SQL query
            
        Returns:
            List[str]: List of syntax issues
        """
        issues = []
        
        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            issues.append("Unbalanced parentheses")
        
        # Check for proper SELECT statement
        if not ( sql.lower().startswith('select') or
                sql.lower().startswith('with') or
                sql.lower().startswith('create') or
                sql.lower().startswith('insert') or
                sql.lower().startswith('update') or
                sql.lower().startswith('delete') or
                sql.lower().startswith('drop') or
                sql.lower().startswith('alter') or
                sql.lower().startswith('describe') or
                sql.lower().startswith('show') or
                sql.lower().startswith('explain') or
                sql.lower().startswith('help') or
                sql.lower().startswith('use') or
                sql.lower().startswith('set') or
                sql.lower().startswith('use')
            ):
            issues.append(f"RAG SQL Query - {sql.split()[0]} -  not a reserved SQL syntax operation word.")
        
        # Check for proper FROM clause
        if 'from' not in sql.lower():
            issues.append("Missing FROM clause")
        
        return issues
    
    def get_validated_plan(self, sql: str) -> Optional[Dict[str, Any]]:
        """Get a previously validated SQL plan.
        
        Args:
            sql (str): The SQL query
            
        Returns:
            Optional[Dict[str, Any]]: The validated plan if found, None otherwise
        """
        cleaned_sql = self.clean_sql_plan(sql)
        return self.validated_plans.get(cleaned_sql)
