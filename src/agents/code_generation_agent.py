# src/agents/code_generation_agent.py
"""
Code Generation Agent - Converts transformation plans into executable pandas code
Generates efficient, readable, and well-documented pandas transformations
"""

from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from src.core.data_schema import GraphState, TransformationPlan, TransformationStep, OperationType
from src.core.llm_config import get_llm
import re

# Initialize LLM
llm = get_llm("llama3-8b-8192", temperature=0.1)  # Lower temperature for consistent code

# === CODE GENERATION TEMPLATES ===

PANDAS_CODE_TEMPLATES = {
    OperationType.STANDARDIZE_COLUMN_NAME: """
# Standardize column name: {column_name}
df = df.rename(columns={{'{column_name}': '{new_name}'}})
print(f"✅ Renamed column '{column_name}' to '{new_name}'")
""",

    OperationType.HANDLE_NULLS: """
# Handle missing values in: {column_name}
{null_handling_code}
print(f"✅ Handled missing values in '{column_name}': {strategy}")
""",

    OperationType.CONVERT_TO_DATETIME: """
# Convert to datetime: {column_name}
try:
    df['{column_name}'] = pd.to_datetime(df['{column_name}'], {datetime_params})
    print(f"✅ Converted '{column_name}' to datetime")
except Exception as e:
    print(f"⚠️ Warning: Could not convert '{column_name}' to datetime: {{e}}")
""",

    OperationType.CONVERT_TO_NUMERIC: """
# Convert to numeric: {column_name}
try:
    {numeric_conversion_code}
    print(f"✅ Converted '{column_name}' to numeric")
except Exception as e:
    print(f"⚠️ Warning: Could not convert '{column_name}' to numeric: {{e}}")
""",

    OperationType.REMOVE_DUPLICATES: """
# Remove duplicate rows
initial_rows = len(df)
df = df.drop_duplicates({duplicate_params})
removed_rows = initial_rows - len(df)
print(f"✅ Removed {{removed_rows}} duplicate rows")
""",

    OperationType.CONVERT_CASE: """
# Convert text case: {column_name}
df['{column_name}'] = df['{column_name}'].str.{case_method}()
print(f"✅ Converted '{column_name}' to {case_type} case")
""",

    OperationType.NORMALIZE: """
# Normalize column: {column_name}
from sklearn.preprocessing import {normalizer_type}
{normalization_code}
print(f"✅ Normalized '{column_name}' using {normalizer_type}")
""",

    OperationType.FILTER_ROWS: """
# Filter rows: {column_name}
initial_rows = len(df)
df = df[{filter_condition}]
filtered_rows = initial_rows - len(df)
print(f"✅ Filtered {{filtered_rows}} rows based on {column_name}")
"""
}

# === CODE GENERATION PROMPTS ===

CODE_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are an expert Python/Pandas code generator. Your task is to convert transformation plans into clean, efficient, executable pandas code.

    **CRITICAL REQUIREMENTS:**
    1. Generate ONLY valid Python/pandas code
    2. Include proper error handling with try/except blocks
    3. Add informative print statements for each operation
    4. Use Brazilian data conventions when specified (comma decimals, date formats)
    5. Code must be production-ready and well-commented
    6. Handle edge cases gracefully
    7. Import required libraries at the top

    **Code Style Guidelines:**
    - Use clear variable names
    - Add comments explaining complex operations
    - Include progress indicators with print statements
    - Handle errors gracefully without stopping execution
    - Follow PEP 8 style guidelines

    **For Brazilian Data:**
    - Use locale='pt_BR' when relevant
    - Handle comma decimal separators (1234,56 → 1234.56)
    - Support DD/MM/YYYY date formats
    - Remove Brazilian currency symbols (R$)

    Generate the complete pandas transformation code based on the provided plan.
    """),
    ("user", """
    Transformation Plan:
    {transformation_plan}

    Data Schema Context:
    {data_schema}

    Generate complete, executable pandas code to implement all transformations.
    Include proper imports, error handling, and progress tracking.
    """)
])

# === CODE GENERATION FUNCTIONS ===

def generate_snake_case_name(column_name: str) -> str:
    """
    Convert column name to snake_case following Brazilian conventions
    """
    # Handle common Brazilian patterns
    name = column_name.lower()
    
    # Replace common patterns
    replacements = {
        ' - ': '_',
        ' da ': '_',
        ' de ': '_',
        ' do ': '_',
        ' das ': '_',
        ' dos ': '_',
        ' ': '_',
        '-': '_',
        '(': '',
        ')': '',
        '/': '_',
        '\\': '_',
        '.': '',
        ',': '',
        ':': ''
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remove multiple underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    return name

def generate_null_handling_code(step: TransformationStep) -> Dict[str, str]:
    """
    Generate code for handling null values based on strategy
    Returns a dictionary with 'code' and 'strategy_desc'
    """
    column = step.column_name
    strategy = step.params.get('strategy', 'fill_zero')
    value = step.params.get('value', 'N/A')
    
    code = ""
    strategy_desc = ""

    if strategy == 'fill_zero':
        code = f"df['{column}'] = df['{column}'].fillna(0)"
        strategy_desc = "filled with 0"
    elif strategy == 'fill_custom':
        code = f"df['{column}'] = df['{column}'].fillna('{value}')"
        strategy_desc = f"filled with '{value}'"
    elif strategy == 'fill_mean':
        code = f"df['{column}'] = df['{column}'].fillna(df['{column}'].mean())"
        strategy_desc = "filled with mean"
    elif strategy == 'fill_median':
        code = f"df['{column}'] = df['{column}'].fillna(df['{column}'].median())"
        strategy_desc = "filled with median"
    elif strategy == 'fill_mode':
        code = f"df['{column}'] = df['{column}'].fillna(df['{column}'].mode()[0] if not df['{column}'].mode().empty else 'Unknown')"
        strategy_desc = "filled with mode"
    elif strategy == 'drop_rows':
        code = f"df = df.dropna(subset=['{column}'])"
        strategy_desc = "rows dropped"
    else: # Default fallback
        code = f"df['{column}'] = df['{column}'].fillna('N/A')"
        strategy_desc = "filled with 'N/A'"
        
    return {"code": code, "strategy_desc": strategy_desc}


def generate_datetime_conversion_code(step: TransformationStep) -> str:
    """
    Generate code for datetime conversion with Brazilian format support
    """
    column = step.column_name
    format_type = step.params.get('format', 'auto')
    
    if format_type == 'auto':
        return f"infer=True, dayfirst=True"  # Brazilian date format (DD/MM/YYYY)
    elif format_type == 'br_format':
        return f"format='%d/%m/%Y', dayfirst=True"
    elif format_type == 'iso_format':
        return f"format='%Y-%m-%d'"
    else:
        return f"infer=True, dayfirst=True"

def generate_numeric_conversion_code(step: TransformationStep) -> str:
    """
    Generate code for numeric conversion with Brazilian decimal support
    """
    column = step.column_name
    decimal_sep = step.params.get('decimal_separator', 'dot')
    remove_currency = step.params.get('remove_currency', False)
    
    code_lines = []
    
    if remove_currency:
        code_lines.append(f"    # Remove currency symbols")
        code_lines.append(f"    df['{column}'] = df['{column}'].astype(str).str.replace(r'[R$€£¥₹]', '', regex=True)")
    
    if decimal_sep == 'comma':
        code_lines.append(f"    # Convert Brazilian decimal format (comma to dot)")
        code_lines.append(f"    df['{column}'] = df['{column}'].astype(str).str.replace(',', '.')")
    
    code_lines.append(f"    # Convert to numeric")
    code_lines.append(f"    df['{column}'] = pd.to_numeric(df['{column}'], errors='coerce')")
    
    return '\n'.join(code_lines)

def generate_step_code(step: TransformationStep) -> str:
    """
    Generate pandas code for a single transformation step
    """
    op_type = step.operation_type
    column = step.column_name
    
    if op_type == OperationType.STANDARDIZE_COLUMN_NAME:
        new_name = generate_snake_case_name(column)
        return PANDAS_CODE_TEMPLATES[op_type].format(
            column_name=column,
            new_name=new_name
        )
    
    elif op_type == OperationType.HANDLE_NULLS:
            # 1. Call the updated function, which now returns a dictionary
            null_handling_data = generate_null_handling_code(step)
            
            # 2. Get the null handling code
            null_code = null_handling_data["code"]
            
            # 3. Get the strategy description to use in the print statement
            strategy_for_print = null_handling_data["strategy_desc"]
            
            # 4. Format the template with the correct information
            return PANDAS_CODE_TEMPLATES[op_type].format(
                column_name=column,
                null_handling_code=null_code,
                # Pass the strategy description to the {strategy} placeholder in the template
                strategy=strategy_for_print 
            )
    
    elif op_type == OperationType.CONVERT_TO_DATETIME:
        datetime_params = generate_datetime_conversion_code(step)
        return PANDAS_CODE_TEMPLATES[op_type].format(
            column_name=column,
            datetime_params=datetime_params
        )
    
    elif op_type == OperationType.CONVERT_TO_NUMERIC:
        numeric_code = generate_numeric_conversion_code(step)
        return PANDAS_CODE_TEMPLATES[op_type].format(
            column_name=column,
            numeric_conversion_code=numeric_code
        )
    
    elif op_type == OperationType.REMOVE_DUPLICATES:
        if column == "all_columns":
            duplicate_params = ""
        else:
            duplicate_params = f"subset=['{column}']"
        return PANDAS_CODE_TEMPLATES[op_type].format(
            duplicate_params=duplicate_params
        )
    
    elif op_type == OperationType.CONVERT_CASE:
        case_method = step.params.get('case', 'lower')
        case_type = step.params.get('case', 'lower')
        return PANDAS_CODE_TEMPLATES[op_type].format(
            column_name=column,
            case_method=case_method,
            case_type=case_type
        )
    
    else:
        # Generic fallback
        return f"""
# {op_type.value}: {column}
# TODO: Implement {op_type.value} operation for column '{column}'
print(f"⚠️ Operation {op_type.value} not yet implemented for '{column}'")
"""

def generate_complete_pipeline_code(transformation_plan: TransformationPlan, file_path: str) -> str:
    """
    Generate complete pandas pipeline code from transformation plan
    """
    code_parts = []
    
    # Header and imports
    code_parts.append("""#!/usr/bin/env python3
\"\"\"
Generated Pandas Data Cleaning Pipeline
Auto-generated from transformation plan
\"\"\"

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "{file_path}"
OUTPUT_FILE = "data/output_parquet/cleaned_data.parquet"

def main():
    \"\"\"Execute the complete data cleaning pipeline\"\"\"
    
    print("🚀 Starting Data Cleaning Pipeline")
    print("=" * 50)
    
    # Load data
    print(f"📂 Loading data from {{INPUT_FILE}}")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"✅ Loaded {{len(df):,}} rows and {{len(df.columns)}} columns")
    except Exception as e:
        print(f"❌ Error loading data: {{e}}")
        return
    
    # Display initial info
    print(f"\\n📊 Initial Data Info:")
    print(f"   - Shape: {{df.shape}}")
    print(f"   - Memory usage: {{df.memory_usage(deep=True).sum() / 1024**2:.2f}} MB")
    print(f"   - Missing values: {{df.isnull().sum().sum():,}}")
    
    original_shape = df.shape""".format(file_path=file_path))
    
    # Add transformation steps
    code_parts.append("\n    # === DATA TRANSFORMATIONS ===")
    
    for i, step in enumerate(transformation_plan.transformation_steps, 1):
        code_parts.append(f"\n    # Step {i}: {step.operation_type.value}")
        code_parts.append(f"    # {step.justification}")
        
        step_code = generate_step_code(step)
        # Indent the step code
        indented_code = '\n'.join('    ' + line if line.strip() else '' 
                                 for line in step_code.split('\n'))
        code_parts.append(indented_code)
    
    # Footer
    code_parts.append("""
    # === PIPELINE COMPLETION ===
    
    # Final data info
    print(f"\\n📊 Final Data Info:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"   - Missing values: {df.isnull().sum().sum():,}")
    
    # Save cleaned data
    print(f"\\n💾 Saving cleaned data to {OUTPUT_FILE}")
    try:
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_FILE, index=False)
        print(f"✅ Data saved successfully")
    except Exception as e:
        print(f"❌ Error saving data: {e}")
    
    # Summary
    print(f"\\n🏁 Pipeline Complete!")
    print("=" * 50)
    print(f"📋 Summary:")
    print(f"   - Original shape: {original_shape}")
    print(f"   - Final shape: {df.shape}")
    print(f"   - Rows changed: {original_shape[0] - df.shape[0]:+,}")
    print(f"   - Columns changed: {original_shape[1] - df.shape[1]:+,}")
    print(f"   - Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()""")
    
    return '\n'.join(code_parts)

# === MAIN CODE GENERATION NODE ===

def code_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate executable pandas code from the approved transformation plan
    
    Args:
        state: Current graph state containing transformation_plan, file_path, etc.
        
    Returns:
        Updated state with generated_code
    """
    transformation_plan: TransformationPlan = state.get("transformation_plan")
    file_path: str = state.get("file_path", "data/input.csv")
    
    if not transformation_plan:
        return {
            "generated_code": None,
            "error_message": "No transformation plan available for code generation"
        }
    
    print("\n--- Starting Code Generation ---")
    print(f"📋 Generating code for {len(transformation_plan.transformation_steps)} transformation steps")
    
    try:
        # Generate the complete pipeline code
        generated_code = generate_complete_pipeline_code(transformation_plan, file_path)
        
        print("✅ Code generation completed successfully")
        print(f"📝 Generated {len(generated_code.split('\n'))} lines of code")
        
        # Create organized directory structure
        import os
        from datetime import datetime
        
        # Create directories
        pipelines_dir = "pipelines"
        generated_dir = os.path.join(pipelines_dir, "generated")
        os.makedirs(generated_dir, exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(generated_dir, f"pipeline_{base_filename}_{timestamp}.py")
        
        # Save the generated code to organized location
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(generated_code)
            print(f"💾 Code saved to {output_file}")
            
            # Also create a 'latest' version for easy access
            latest_file = os.path.join(generated_dir, f"pipeline_{base_filename}_latest.py")
            with open(latest_file, 'w', encoding='utf-8') as f:
                f.write(generated_code)
            print(f"📄 Latest version saved to {latest_file}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save code to file: {e}")
            output_file = "generated_pipeline.py"  # Fallback to root
        
        return {
            "generated_code": generated_code,
            "code_file_path": output_file,
            "latest_code_path": latest_file if 'latest_file' in locals() else None,
            "error_message": None
        }
        
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        return {
            "generated_code": None,
            "error_message": f"Code generation error: {e}"
        }

# === TEST FUNCTION ===

def test_code_generation():
    """Test the code generation with a sample transformation plan"""
    from src.core.data_schema import TransformationStep, OperationType
    
    # Create sample transformation plan
    sample_steps = [
        TransformationStep(
            operation_type=OperationType.STANDARDIZE_COLUMN_NAME,
            column_name="Regiao - Sigla",
            params={},
            justification="Standardize column name",
            expected_outcome="Snake case column name"
        ),
        TransformationStep(
            operation_type=OperationType.CONVERT_TO_DATETIME,
            column_name="Data da Coleta",
            params={"format": "auto"},
            justification="Convert to datetime",
            expected_outcome="Datetime column"
        ),
        TransformationStep(
            operation_type=OperationType.CONVERT_TO_NUMERIC,
            column_name="Valor de Venda",
            params={"decimal_separator": "comma", "remove_currency": True},
            justification="Convert to numeric with Brazilian format",
            expected_outcome="Numeric column"
        )
    ]
    
    sample_plan = TransformationPlan(
        initial_data_overview_summary="Sample dataset",
        transformation_steps=sample_steps,
        final_output_format="parquet",
        overall_summary="Sample transformation plan",
        requires_confirmation=True
    )
    
    # Generate code
    code = generate_complete_pipeline_code(sample_plan, "data/sample.csv")
    print("Generated code:")
    print("=" * 50)
    print(code)

if __name__ == "__main__":
    test_code_generation()