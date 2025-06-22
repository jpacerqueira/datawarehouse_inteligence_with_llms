from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum

class ColumnImportance(Enum):
    """Enum to represent the importance of a column for vector RAG embeddings."""
    IMPORTANT = "important"  # Column should be learned for vector RAG embeddings
    NOT_IMPORTANT = "not_important"  # Column should not be learned for vector RAG embeddings

@dataclass
class ColumnReference:
    """Reference structure for a column's vector RAG embedding importance."""
    table_name: str
    column_name: str
    importance: ColumnImportance
    description: str = ""

class VectorRAGEmbeddingsReference:
    """Reference structure for vector RAG embeddings importance across tables and columns."""
    
    def __init__(self):
        # Initialize with the hardcoded list of columns that should not be learned
        self._not_important_columns: Set[Tuple[str, str]] = {
            ("artefacts", "OP"),
            ("artefacts", "DMS_TIMESTAMP"),
            ("artefact_schedule_dates", "OP"),
            ("artefact_schedule_dates", "DMS_TIMESTAMP"),
            ("bank_accounts", "OP"),
            ("bank_accounts", "DMS_TIMESTAMP"),
            ("bank_payments", "OP"),
            ("bank_payments", "DMS_TIMESTAMP"),
            ("bank_payment_types", "OP"),
            ("bank_payment_types", "DMS_TIMESTAMP"),
            ("bank_receipts", "OP"),
            ("bank_receipts", "DMS_TIMESTAMP"),
            ("bank_receipt_types", "OP"),
            ("bank_receipt_types", "DMS_TIMESTAMP"),
            ("businesses", "OP"),
            ("businesses", "DMS_TIMESTAMP"),
            ("contacts", "OP"),
            ("contacts", "DMS_TIMESTAMP"),
            ("countries", "OP"),
            ("countries", "DMS_TIMESTAMP"),
            ("currencies", "OP"),
            ("currencies", "DMS_TIMESTAMP"),
            ("ledger_accounts", "OP"),
            ("ledger_accounts", "DMS_TIMESTAMP"),
            ("ledger_account_types", "OP"),
            ("ledger_account_types", "DMS_TIMESTAMP"),
            ("ledger_entries", "OP"),
            ("ledger_entries", "DMS_TIMESTAMP"),
            ("payments", "OP"),
            ("payments", "DMS_TIMESTAMP"),
            ("payment_artefacts", "OP"),
            ("payment_artefacts", "DMS_TIMESTAMP"),
            ("payment_types", "OP"),
            ("payment_types", "DMS_TIMESTAMP"),
            ("tax_returns", "OP"),
            ("tax_returns", "DMS_TIMESTAMP"),
            ("transactions", "OP"),
            ("transactions", "DMS_TIMESTAMP"),
            ("transaction_types", "OP"),
            ("transaction_types", "DMS_TIMESTAMP"),
            ("cash_flow_by_account", "OP"),
            ("cash_flow_by_account", "DMS_TIMESTAMP"),
            ("cash_flow_seasonality", "OP"),
            ("cash_flow_seasonality", "DMS_TIMESTAMP"),
            ("comprehensive_cash_dashboard", "OP"),
            ("comprehensive_cash_dashboard", "DMS_TIMESTAMP"),
            ("current_bank_balances", "OP"),
            ("current_bank_balances", "DMS_TIMESTAMP"),
            ("current_cash_position", "OP"),
            ("current_cash_position", "DMS_TIMESTAMP"),
            ("customer_segment_analysis", "OP"),
            ("customer_segment_analysis", "DMS_TIMESTAMP"),
            ("invoice_payment_timeline", "OP"),
            ("invoice_payment_timeline", "DMS_TIMESTAMP"),
            ("multi_currency_forecast", "OP"),
            ("multi_currency_forecast", "DMS_TIMESTAMP"),
            ("payment_method_analysis", "OP"),
            ("payment_method_analysis", "DMS_TIMESTAMP"),
            ("payment_patterns_analysis", "OP"),
            ("payment_patterns_analysis", "DMS_TIMESTAMP"),
            ("projected_cash_flow_90_days", "OP"),
            ("projected_cash_flow_90_days", "DMS_TIMESTAMP"),
            ("seasonal_cash_flow_analysis", "OP"),
            ("seasonal_cash_flow_analysis", "DMS_TIMESTAMP"),
            ("transaction_type_analysis", "OP"),
            ("transaction_type_analysis", "DMS_TIMESTAMP"),
            ("upcoming_accounts_receivable", "OP"),
            ("upcoming_accounts_receivable", "DMS_TIMESTAMP"),
        }
        
        # Cache for column references
        self._column_references: Dict[Tuple[str, str], ColumnReference] = {}
        
    def get_column_importance(self, table_name: str, column_name: str) -> ColumnImportance:
        """Get the importance of a column for vector RAG embeddings.
        
        Args:
            table_name (str): Name of the table
            column_name (str): Name of the column
            
        Returns:
            ColumnImportance: The importance level of the column
        """
        return (ColumnImportance.NOT_IMPORTANT 
                if (table_name, column_name) in self._not_important_columns 
                else ColumnImportance.IMPORTANT)
    
    def get_column_reference(self, table_name: str, column_name: str) -> ColumnReference:
        """Get the reference information for a column.
        
        Args:
            table_name (str): Name of the table
            column_name (str): Name of the column
            
        Returns:
            ColumnReference: The reference information for the column
        """
        key = (table_name, column_name)
        if key not in self._column_references:
            importance = self.get_column_importance(table_name, column_name)
            description = (
                "Column excluded from vector RAG embeddings learning"
                if importance == ColumnImportance.NOT_IMPORTANT
                else "Column included in vector RAG embeddings learning"
            )
            self._column_references[key] = ColumnReference(
                table_name=table_name,
                column_name=column_name,
                importance=importance,
                description=description
            )
        return self._column_references[key]
    
    def add_not_important_column(self, table_name: str, column_name: str) -> None:
        """Add a column to the list of columns that should not be learned.
        
        Args:
            table_name (str): Name of the table
            column_name (str): Name of the column
        """
        self._not_important_columns.add((table_name, column_name))
        # Clear the cache for this column
        key = (table_name, column_name)
        if key in self._column_references:
            del self._column_references[key]
    
    def remove_not_important_column(self, table_name: str, column_name: str) -> None:
        """Remove a column from the list of columns that should not be learned.
        
        Args:
            table_name (str): Name of the table
            column_name (str): Name of the column
        """
        self._not_important_columns.discard((table_name, column_name))
        # Clear the cache for this column
        key = (table_name, column_name)
        if key in self._column_references:
            del self._column_references[key]
    
    def get_all_not_important_columns(self) -> List[Tuple[str, str]]:
        """Get all columns that should not be learned.
        
        Returns:
            List[Tuple[str, str]]: List of (table_name, column_name) tuples
        """
        return sorted(list(self._not_important_columns))
    
    def clear_cache(self) -> None:
        """Clear the column references cache."""
        self._column_references.clear() 