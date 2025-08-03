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
    import numpy as np
    
    dtype = str(series.dtype)
    is_nullable = series.isnull().any()
    unique_values = series.nunique() if series.dtype == 'object' or series.nunique() < 50 else None
    
    # Convert unique_values to Python int if it's numpy
    if unique_values is not None and isinstance(unique_values, np.integer):
        unique_values = int(unique_values)
    
    top_values = None
    if series.dtype == 'object' or series.nunique() < 20:
        top_values_counts = series.value_counts(dropna=False).head(5)
        top_values = [{"value": str(idx), "count": int(count)} for idx, count in top_values_counts.items()]

    min_val, max_val = None, None
    try:
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            min_val = series.min()
            max_val = series.max()
            
            # Convert numpy types to Python native types
            if min_val is not None:
                if isinstance(min_val, np.integer):
                    min_val = int(min_val)
                elif isinstance(min_val, np.floating):
                    min_val = float(min_val)
                elif pd.api.types.is_datetime64_any_dtype(series):
                    min_val = str(min_val)
                    
            if max_val is not None:
                if isinstance(max_val, np.integer):
                    max_val = int(max_val)
                elif isinstance(max_val, np.floating):
                    max_val = float(max_val)
                elif pd.api.types.is_datetime64_any_dtype(series):
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

def _detect_mixed_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect columns with mixed data types.
    
    Args:
        df: A Pandas DataFrame object.
        
    Returns:
        Dictionary with column names as keys and type information as values.
    """
    mixed_types = {}
    
    for col in df.columns:
        # Get unique types in the column
        types = set()
        for value in df[col].dropna():
            types.add(type(value).__name__)
        
        # If more than one type is found, it's mixed
        if len(types) > 1:
            mixed_types[col] = {
                "types": list(types),
                "sample_values": df[col].dropna().head(3).tolist()
            }
    
    return mixed_types

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
    
    # Convert numpy types to Python native types for serialization
    import numpy as np
    
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert the schema to a dict and handle numpy types
    schema_dict = temp_df_schema.model_dump()
    schema_dict = convert_numpy_types(schema_dict)
    
    # Convert back to JSON string
    import json
    schema_json = json.dumps(schema_dict, indent=2)
    
    llm_summary_output = _get_summary_chain().invoke({"schema_json": schema_json})
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
    Versão mais robusta com melhor tratamento de tipos mistos
    """
    file_path = state.get("file_path")
    source_type = state.get("source_type", "csv")

    if not file_path:
        raise ValueError("File path not provided in graph state.")

    df = None
    load_info = {"warnings": [], "method_used": ""}
    
    if source_type == "csv":
        # Lista de estratégias de leitura
        read_strategies = [
            # Estratégia 1: Padrão com low_memory=False
            {
                "params": {"low_memory": False, "na_values": ['', 'NULL', 'null', 'NaN', 'nan', 'N/A', '#N/A']},
                "name": "padrão"
            },
            # Estratégia 2: Com separador específico
            {
                "params": {"sep": ",", "low_memory": False, "na_values": ['', 'NULL', 'null', 'NaN', 'nan', 'N/A']},
                "name": "sep=','"
            },
            # Estratégia 3: Auto-detect com engine python
            {
                "params": {"sep": None, "engine": "python", "low_memory": False},
                "name": "auto-detect"
            },
            # Estratégia 4: Fallback - tudo como string
            {
                "params": {"dtype": str, "low_memory": False},
                "name": "dtype=str"
            }
        ]
        
        last_error = None
        for strategy in read_strategies:
            try:
                df = pd.read_csv(file_path, **strategy["params"])
                load_info["method_used"] = strategy["name"]
                print(f"✅ CSV carregado ({strategy['name']}): {df.shape[0]} linhas, {df.shape[1]} colunas")
                break
            except Exception as e:
                last_error = e
                continue
        
        if df is None:
            raise ValueError(f"Não foi possível carregar CSV {file_path}. Último erro: {last_error}")
            
    elif source_type == "json":
        try:
            df = pd.read_json(file_path)
            load_info["method_used"] = "json_padrão"
        except Exception as e:
            raise ValueError(f"Error reading JSON file {file_path}: {e}")
            
    elif source_type == "excel":
        try:
            df = pd.read_excel(file_path)
            load_info["method_used"] = "excel_padrão"
        except Exception as e:
            raise ValueError(f"Error reading Excel file {file_path}: {e}")
    else:
        raise ValueError(f"Data source type '{source_type}' not supported yet.")

    if df is None:
        raise ValueError("Could not load DataFrame.")

    # Detectar e reportar problemas de tipos
    mixed_type_info = _detect_mixed_types(df)
    if mixed_type_info:
        load_info["warnings"].append(f"Colunas com tipos mistos: {list(mixed_type_info.keys())}")
        print(f"⚠️  {len(mixed_type_info)} colunas com tipos mistos detectadas")

    df_schema = _analyze_dataframe(df)
    
    # Adicionar info de carregamento ao schema
    df_schema.load_info = load_info

    return {"df_schema": df_schema}
