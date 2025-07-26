# src/agents/data_ingestion_agent.py

import pandas as pd
import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Core module imports (now as functions or global instances if needed)
from src.core.data_schema import ColumnSchema, DataFrameSchema
from src.core.llm_config import get_llm # Gets an LLM instance

# We'll create the LLM instance lazily when needed, not at module import time
def _get_llm_instance():
    """Get LLM instance when needed, ensuring environment is loaded."""
    return get_llm("llama3-8b-8192", temperature=0.1)

# Prompt and chain for dataset textual summary
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an experienced and concise data analyst. Given the schema analysis and some statistics of a Pandas DataFrame, generate a clear and direct textual summary. Highlight important characteristics, such as key columns, potential issues (nulls, mixed types), and initial standardization suggestions, if applicable."),
    ("user", "Here is the DataFrame schema:\n{schema_json}\n\nBased on this information, provide a descriptive summary of the dataset, focusing on insights for transformations. Be concise and direct. Output in English.")
])

def _get_summary_chain():
    """Get the summary chain with LLM instance created when needed."""
    return SUMMARY_PROMPT | _get_llm_instance()

#
def _infer_column_schema(series: pd.Series) -> ColumnSchema:
    """Infers the schema for a single Pandas Series column.
    Args:
        series: A Pandas Series object representing a column of the DataFrame.

    Returns:
        A ColumnSchema object representing the schema of the column.
    """
    dtype = str(series.dtype)
    is_nullable = series.isnull().any()
    unique_values = series.nunique() if series.dtype == 'object' or series.nunique() < 50 else None
    top_values = None
    if series.dtype == 'object' or series.nunique() < 20:
        top_values_counts = series.value_counts(dropna=False).head(5)
        top_values = [{"value": str(idx), "count": int(count)} for idx, count in top_values_counts.items()]

    min_val, max_val = None, None
    try:
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            min_val = series.min()
            max_val = series.max()
            if pd.api.types.is_datetime64_any_dtype(series) and min_val is not None:
                min_val = str(min_val)
            if pd.api.types.is_datetime64_any_dtype(series) and max_val is not None:
                max_val = str(max_val)
        # Handle cases where min/max might not be directly comparable (e.g., mixed types in object column)
        elif series.dtype == 'object':
            pass # No min/max for object dtype unless explicitly converted
    except TypeError:
        pass # Can fail if there are mixed types, ignore min/max in this case

    return ColumnSchema(
        name=series.name,
        dtype=dtype,
        is_nullable=is_nullable,
        unique_values=unique_values,
        top_values=top_values,
        min_value=min_val,
        max_value=max_val,
    )

def _analyze_dataframe(df: pd.DataFrame) -> DataFrameSchema:
    """Analyzes the DataFrame and returns a DataFrameSchema object.
    Args:
        df: A Pandas DataFrame object representing the dataset.

    Returns:
        A DataFrameSchema object representing the schema of the dataset.
    """
    num_rows = len(df)
    num_columns = len(df.columns)
    columns_schema = [_infer_column_schema(df[col]) for col in df.columns]

    # Generates the textual summary using the LLM
    temp_df_schema = DataFrameSchema(
        num_rows=num_rows,
        num_columns=num_columns,
        columns=columns_schema,
        overview_summary="Placeholder for LLM to generate." # Placeholder that will be filled
    )
    llm_summary_output = _get_summary_chain().invoke({"schema_json": temp_df_schema.model_dump_json(indent=2)})
    overview_summary = llm_summary_output.content

    return DataFrameSchema(
        num_rows=num_rows,
        num_columns=num_columns,
        columns=columns_schema,
        overview_summary=overview_summary,
    )

# Main "agent" function - will be a node in Langgraph
def data_ingestion_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Langgraph node for data ingestion and analysis.
    Receives file_path and source_type from state and returns df_schema.

    Args:
        state: A dictionary containing the file_path and source_type.

    Returns:
        A dictionary containing the df_schema.
    """
    file_path = state.get("file_path")
    source_type = state.get("source_type", "csv") # Default to CSV if not specified

    if not file_path:
        raise ValueError("File path not provided in graph state.")

    df = None
    if source_type == "csv":
        try:
            # Try to read with comma delimiter first
            df = pd.read_csv(file_path)
        except Exception as e:
            # If that fails, try with semicolon delimiter
            try:
                df = pd.read_csv(file_path, sep=';')
            except Exception as e2:
                # If both fail, try to auto-detect the delimiter
                try:
                    df = pd.read_csv(file_path, sep=None, engine='python')
                except Exception as e3:
                    raise ValueError(f"Error reading CSV file {file_path}. Tried comma, semicolon, and auto-detection. Errors: {e}, {e2}, {e3}")
    elif source_type == "json":
        try:
            df = pd.read_json(file_path)
        except Exception as e:
            raise ValueError(f"Error reading JSON file {file_path}: {e}")
    elif source_type == "excel":
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"Error reading Excel file {file_path}: {e}")
    else:
        raise ValueError(f"Data source type '{source_type}' not supported yet.")

    if df is None:
        raise ValueError("Could not load DataFrame.")

    df_schema = _analyze_dataframe(df)

    # Returns a dictionary that will be used to update the graph state
    return {"df_schema": df_schema}
