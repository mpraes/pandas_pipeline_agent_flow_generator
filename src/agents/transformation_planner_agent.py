# src/agents/transformation_planner_agent.py

import re           
import traceback
import json
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

from src.core.data_schema import GraphState, TransformationPlan, DataFrameSchema, TransformationStep, OperationType, SearchQueries, UserApproval
from src.core.llm_config import get_llm
from src.core.tavily_utils import search_tavily

# IMPORT FOR ADVANCED RAG
try:
    from src.utils.advanced_rag_system import create_rag_system
    ADVANCED_RAG_AVAILABLE = True
    print("✅ Advanced RAG system available")
except ImportError as e:
    ADVANCED_RAG_AVAILABLE = False
    print(f"⚠️ Advanced RAG not available: {e}")
    print("   Falling back to Tavily search")

import os

# Initialize LLM for direct use in chains where needed.
llm = get_llm("llama3-8b-8192", temperature=0.3)

# === ALL NECESSARY AUXILIARY FUNCTIONS ===

def detect_data_characteristics(data_overview: str) -> Dict[str, bool]:
    """
    Dynamically detects data characteristics from the overview to guide search strategy.
    """
    overview_lower = data_overview.lower()
    
    characteristics = {
        "has_datetime": any(term in overview_lower for term in ["date", "time", "datetime", "timestamp"]),
        "has_text_heavy": any(term in overview_lower for term in ["object", "string", "text", "categorical"]),
        "has_numerical": any(term in overview_lower for term in ["float", "int", "numeric", "number"]),
        "has_missing_data": any(term in overview_lower for term in ["missing", "null", "nan", "none"]),
        "is_large_dataset": any(term in overview_lower for term in ["thousand", "million", "large", "rows"]),
        "has_encoding_issues": any(term in overview_lower for term in ["encoding", "utf", "unicode", "character"]),
        "has_duplicates": "duplicate" in overview_lower,
        "has_mixed_types": "mixed" in overview_lower or "inconsistent" in overview_lower
    }
    
    return characteristics

def adjust_relevance_threshold_dynamically(queries: List[str], data_characteristics: Dict[str, bool]) -> float:
    """
    Dynamically adjusts relevance threshold based on query specificity and data characteristics.
    """
    base_threshold = 6.0  # Start with lower base threshold
    
    # Count technical pandas terms in queries
    technical_terms = [
        "pandas", "dataframe", "fillna", "astype", "groupby", "merge", 
        "to_datetime", "drop_duplicates", "rename", "categorical", "dtype",
        "preprocessing", "cleaning", "standardization"
    ]
    
    technical_queries = sum(
        1 for query in queries 
        if any(term in query.lower() for term in technical_terms)
    )
    
    specificity_ratio = technical_queries / len(queries) if queries else 0
    
    # More aggressive threshold reduction
    if specificity_ratio > 0.8:
        base_threshold -= 2.5  # Very technical queries
    elif specificity_ratio > 0.5:
        base_threshold -= 1.5  # Moderately technical
    elif specificity_ratio > 0.2:
        base_threshold -= 1.0  # Some technical terms
    
    # Adjust based on data complexity
    complexity_indicators = sum([
        data_characteristics.get("has_mixed_types", False),
        data_characteristics.get("has_encoding_issues", False),
        data_characteristics.get("is_large_dataset", False)
    ])
    
    if complexity_indicators >= 2:
        base_threshold -= 1.5  # Complex data needs more permissive search
    elif complexity_indicators >= 1:
        base_threshold -= 1.0
    
    return max(3.0, base_threshold)  # Allow threshold as low as 3.0

def implement_data_engineering_fallbacks(data_characteristics: Dict[str, bool], user_instructions: str = "") -> List[str]:
    """
    Implements fallback strategies focused on core data engineering operations.
    """
    fallback_queries = []
    
    # Core data engineering fallbacks
    fallback_queries.extend([
        "pandas data cleaning fundamentals",
        "pandas data preprocessing steps",
        "pandas data quality best practices",
        "pandas ETL pipeline basics"
    ])
    
    # Characteristic-specific fallbacks
    if data_characteristics.get("has_datetime"):
        fallback_queries.extend([
            "pandas datetime cleaning tutorial",
            "date parsing pandas comprehensive guide"
        ])
    
    if data_characteristics.get("has_text_heavy"):
        fallback_queries.extend([
            "pandas string data cleaning guide",
            "text preprocessing pandas tutorial"
        ])
    
    if data_characteristics.get("has_numerical"):
        fallback_queries.extend([
            "pandas numerical data cleaning",
            "pandas numeric type optimization"
        ])
    
    if data_characteristics.get("has_missing_data"):
        fallback_queries.extend([
            "pandas missing data comprehensive guide",
            "pandas null handling strategies"
        ])
    
    if data_characteristics.get("is_large_dataset"):
        fallback_queries.extend([
            "pandas memory efficient processing",
            "pandas large dataset optimization"
        ])
    
    # General purpose fallbacks
    fallback_queries.extend([
        "pandas cookbook data cleaning",
        "pandas user guide preprocessing",
        "common pandas data issues solutions"
    ])
    
    return fallback_queries

def implement_aggressive_search_strategy(user_instructions: str, data_overview: str) -> List[str]:
    """
    Implements very broad and generic search queries when specific searches fail.
    """
    # Super generic queries that should always find something
    aggressive_queries = [
        "pandas tutorial",
        "pandas basics",
        "pandas documentation",
        "python pandas examples",
        "pandas data cleaning",
        "pandas user guide",
        "pandas cookbook",
        "pandas operations",
        "pandas dataframe",
        "pandas how to"
    ]
    
    return aggressive_queries

def enhance_queries_with_data_engineering_focus(original_queries: List[str], user_instructions: str, data_overview: str) -> List[str]:
    """
    Enhances queries with focus on generic data engineering and cleaning operations.
    """
    enhanced_queries = []
    
    # Core pandas data engineering operation mappings
    data_engineering_mappings = {
        "convert object to datetime": [
            "pandas to_datetime parsing errors",
            "datetime format detection pandas",
            "pandas date parsing best practices"
        ],
        "fillna strategies": [
            "pandas missing data strategies comparison",
            "fillna vs dropna decision criteria",
            "pandas imputation techniques"
        ],
        "missing values": [
            "pandas null value detection patterns",
            "missing data visualization pandas",
            "pandas handle different null representations"
        ],
        "standardize categorical": [
            "pandas string data cleaning",
            "categorical data preprocessing pandas",
            "pandas text normalization techniques"
        ],
        "normalize numerical": [
            "pandas numerical data scaling",
            "data normalization techniques pandas",
            "pandas outlier detection cleaning"
        ],
        "remove duplicates": [
            "pandas duplicate detection strategies",
            "drop_duplicates parameters pandas",
            "identify duplicate patterns pandas"
        ],
        "column names": [
            "pandas column naming conventions",
            "bulk column renaming pandas",
            "clean column names pandas"
        ],
        "data types": [
            "pandas dtype optimization",
            "memory efficient pandas dtypes",
            "pandas type conversion best practices"
        ]
    }
    
    # Add original queries
    for query in original_queries:
        enhanced_queries.append(query)
        
        # Find relevant expansions based on detected patterns
        for pattern, expansions in data_engineering_mappings.items():
            if any(word in query.lower() for word in pattern.split()):
                enhanced_queries.extend(expansions[:2])
    
    # Always add core data engineering fundamentals
    enhanced_queries.extend([
        "pandas data cleaning best practices",
        "data engineering pandas workflow",
        "pandas data validation techniques",
        "pandas data type optimization"
    ])
    
    # Remove duplicates while maintaining order
    seen = set()
    final_queries = []
    for query in enhanced_queries:
        if query not in seen:
            seen.add(query)
            final_queries.append(query)
    
    return final_queries[:18]  # Reasonable limit for search performance

# === ADICIONE ESTA FUNÇÃO APÓS enhance_queries_with_data_engineering_focus() ===

# ADICIONE ESTA FUNÇÃO PARA OTIMIZAR O CONTEXTO:

def optimize_context_for_llm(data_schema: DataFrameSchema, crag_context: str, max_columns: int = 20, max_context_chars: int = 3000) -> Tuple[str, str]:
    """
    Optimize data schema and CRAG context to fit within LLM context limits
    """
    try:
        # 1. Optimize data schema - summarize instead of full JSON
        schema_summary = {
            "total_columns": len(data_schema.columns) if data_schema.columns else 0,
            "overview": data_schema.overview_summary,
            "sample_columns": []
        }
        
        # Include only most important columns
        if data_schema.columns:
            # Prioritize columns with issues (nulls, mixed types, dates, etc.)
            important_columns = []
            for col in data_schema.columns[:max_columns]:
                col_info = {
                    "name": getattr(col, 'name', str(col)),
                    "dtype": getattr(col, 'dtype', 'unknown'),
                    "nullable": getattr(col, 'is_nullable', True)
                }
                important_columns.append(col_info)
            
            schema_summary["sample_columns"] = important_columns
        
        optimized_schema = json.dumps(schema_summary, indent=2)
        
        # 2. Optimize CRAG context - truncate if too long
        optimized_context = crag_context
        if len(crag_context) > max_context_chars:
            # Keep most relevant parts
            context_parts = crag_context.split("\n\n")
            total_length = 0
            kept_parts = []
            
            for part in context_parts:
                if total_length + len(part) <= max_context_chars:
                    kept_parts.append(part)
                    total_length += len(part)
                else:
                    break
            
            optimized_context = "\n\n".join(kept_parts)
            if len(optimized_context) < len(crag_context):
                optimized_context += "\n\n[Context truncated to fit model limits]"
        
        print(f"📊 Context optimization:")
        print(f"   Original schema length: ~{len(data_schema.model_dump_json())}")
        print(f"   Optimized schema length: {len(optimized_schema)}")
        print(f"   Original context length: {len(crag_context)}")
        print(f"   Optimized context length: {len(optimized_context)}")
        
        return optimized_schema, optimized_context
        
    except Exception as e:
        print(f"Error optimizing context: {e}")
        # Fallback: very minimal context
        minimal_schema = json.dumps({
            "overview": getattr(data_schema, 'overview_summary', 'Dataset with multiple columns'),
            "total_columns": len(data_schema.columns) if hasattr(data_schema, 'columns') and data_schema.columns else 0
        })
        minimal_context = crag_context[:1000] + "... [truncated]" if len(crag_context) > 1000 else crag_context
        return minimal_schema, minimal_context

def parse_llm_response_to_json(response_text: str) -> dict:
    """
    Robust parsing of LLM response to extract JSON
    """
    try:
        print(f"📝 Parsing LLM response (length: {len(response_text)})")
        
        # Strategy 1: Direct JSON parsing (entire response)
        try:
            plan_dict = json.loads(response_text.strip())
            if isinstance(plan_dict, dict):
                print("✅ Successfully parsed entire response as JSON")
                return plan_dict
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Find JSON blocks with multiple patterns
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON pattern
            r'\{.*?\}',  # Simple JSON pattern
            r'```json\s*(\{.*?\})\s*```',  # JSON code block
            r'```\s*(\{.*?\})\s*```',  # Generic code block
            r'(?s)\{.*\}',  # Multiline JSON with DOTALL
        ]
        
        for i, pattern in enumerate(json_patterns):
            try:
                matches = re.findall(pattern, response_text, re.DOTALL)
                for match in matches:
                    # Handle tuple results from capture groups
                    json_str = match if isinstance(match, str) else (match[0] if match else "")
                    
                    if json_str.strip():
                        try:
                            plan_dict = json.loads(json_str)
                            if isinstance(plan_dict, dict) and "transformation_steps" in plan_dict:
                                print(f"✅ Successfully parsed JSON using pattern {i+1}")
                                return plan_dict
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Pattern {i+1} failed: {e}")
                continue
        
        # Strategy 3: Manual JSON extraction with safer bounds
        try:
            # Find all potential JSON start/end positions
            json_starts = [i for i, char in enumerate(response_text) if char == '{']
            json_ends = [i for i, char in enumerate(response_text) if char == '}']
            
            if json_starts and json_ends:
                # Try different combinations of start/end positions
                for start_idx in json_starts[:3]:  # Try first 3 opening braces
                    for end_idx in reversed(json_ends[-3:]):  # Try last 3 closing braces
                        if end_idx > start_idx:
                            try:
                                json_str = response_text[start_idx:end_idx + 1]
                                plan_dict = json.loads(json_str)
                                if isinstance(plan_dict, dict):
                                    print("✅ Successfully parsed JSON using manual extraction")
                                    return plan_dict
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            print(f"Manual extraction failed: {e}")
        
        # Strategy 4: Try to construct JSON from key phrases
        try:
            # Look for key transformation step indicators
            if "transformation_steps" in response_text.lower():
                print("🔄 Attempting to construct JSON from text content...")
                
                # Extract basic information
                overview_match = re.search(r'"initial_data_overview_summary":\s*"([^"]*)"', response_text)
                overview = overview_match.group(1) if overview_match else "Dataset overview"
                
                summary_match = re.search(r'"overall_summary":\s*"([^"]*)"', response_text)
                summary = summary_match.group(1) if summary_match else "Data cleaning pipeline"
                
                # Create minimal valid structure
                constructed_dict = {
                    "initial_data_overview_summary": overview,
                    "transformation_steps": [],
                    "final_output_format": "parquet",
                    "overall_summary": summary,
                    "requires_confirmation": True
                }
                
                print("✅ Constructed minimal JSON from text content")
                return constructed_dict
                
        except Exception as e:
            print(f"JSON construction failed: {e}")
        
        print("❌ All JSON parsing strategies failed")
        return None
        
    except Exception as e:
        print(f"❌ Error in parse_llm_response_to_json: {e}")
        return None


    """
    Robust parsing of LLM response to extract JSON
    """
    try:
        print(f"📝 Parsing LLM response (length: {len(response_text)})")
        
        # Strategy 1: Direct JSON parsing (entire response)
        try:
            plan_dict = json.loads(response_text.strip())
            if isinstance(plan_dict, dict):
                print("✅ Successfully parsed entire response as JSON")
                return plan_dict
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Find JSON blocks with multiple patterns
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON pattern
            r'\{.*?\}',  # Simple JSON pattern
            r'```json\s*(\{.*?\})\s*```',  # JSON code block
            r'```\s*(\{.*?\})\s*```',  # Generic code block
            r'(?s)\{.*\}',  # Multiline JSON with DOTALL
        ]
        
        for i, pattern in enumerate(json_patterns):
            try:
                matches = re.findall(pattern, response_text, re.DOTALL)
                for match in matches:
                    # Handle tuple results from capture groups
                    json_str = match if isinstance(match, str) else (match[0] if match else "")
                    
                    if json_str.strip():
                        try:
                            plan_dict = json.loads(json_str)
                            if isinstance(plan_dict, dict) and "transformation_steps" in plan_dict:
                                print(f"✅ Successfully parsed JSON using pattern {i+1}")
                                return plan_dict
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Pattern {i+1} failed: {e}")
                continue
        
        # Strategy 3: Manual JSON extraction with safer bounds
        try:
            # Find all potential JSON start/end positions
            json_starts = [i for i, char in enumerate(response_text) if char == '{']
            json_ends = [i for i, char in enumerate(response_text) if char == '}']
            
            if json_starts and json_ends:
                # Try different combinations of start/end positions
                for start_idx in json_starts[:3]:  # Try first 3 opening braces
                    for end_idx in reversed(json_ends[-3:]):  # Try last 3 closing braces
                        if end_idx > start_idx:
                            try:
                                json_str = response_text[start_idx:end_idx + 1]
                                plan_dict = json.loads(json_str)
                                if isinstance(plan_dict, dict):
                                    print("✅ Successfully parsed JSON using manual extraction")
                                    return plan_dict
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            print(f"Manual extraction failed: {e}")
        
        # Strategy 4: Try to construct JSON from key phrases
        try:
            # Look for key transformation step indicators
            if "transformation_steps" in response_text.lower():
                print("🔄 Attempting to construct JSON from text content...")
                
                # Extract basic information
                overview_match = re.search(r'"initial_data_overview_summary":\s*"([^"]*)"', response_text)
                overview = overview_match.group(1) if overview_match else "Dataset overview"
                
                summary_match = re.search(r'"overall_summary":\s*"([^"]*)"', response_text)
                summary = summary_match.group(1) if summary_match else "Data cleaning pipeline"
                
                # Create minimal valid structure
                constructed_dict = {
                    "initial_data_overview_summary": overview,
                    "transformation_steps": [],
                    "final_output_format": "parquet",
                    "overall_summary": summary,
                    "requires_confirmation": True
                }
                
                print("✅ Constructed minimal JSON from text content")
                return constructed_dict
                
        except Exception as e:
            print(f"JSON construction failed: {e}")
        
        print("❌ All JSON parsing strategies failed")
        return None
        
    except Exception as e:
        print(f"❌ Error in parse_llm_response_to_json: {e}")
        return None

# === PROMPTS AND CHAINS ===
PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are an expert Data Engineering Agent specializing in Pandas data transformations.
        
        **CRITICAL**: You MUST generate a valid JSON object that matches the TransformationPlan schema EXACTLY.
        
        Required fields for each transformation step:
        - operation_type: Must be one of these EXACT values: 'standardize_column_name', 'handle_nulls', 'convert_case', 'change_datatype', 'filter_rows', 'aggregate', 'remove_duplicates', 'derive_column', 'rename_column', 'convert_to_datetime', 'convert_to_numeric', 'fill_nulls', 'normalize', 'feature_engineering', 'data_visualization'
        - column_name: Exact column name from the schema
        - params: Dictionary (can be empty {{}})
        - justification: Clear explanation of why this step is needed
        - expected_outcome: What the result will be
        
        Focus EXCLUSIVELY on data engineering and cleaning operations. NO business logic.
        
        IMPORTANT: Use only the exact operation_type values listed above. Do NOT create new values.
        
        Example of correct format:
        {{
          "initial_data_overview_summary": "Dataset overview...",
          "transformation_steps": [
            {{
              "operation_type": "standardize_column_name",
              "column_name": "Regiao - Sigla",
              "params": {{}},
              "justification": "Column name contains spaces and special characters",
              "expected_outcome": "Standardized column name: regiao_sigla"
            }}
          ],
          "final_output_format": "parquet",
          "overall_summary": "Pipeline summary...",
          "requires_confirmation": true
        }}
        """),
        ("user", """
        DataFrame Schema:
        ```json
        {data_schema_json}
        ```

        User Instructions: {user_instructions}
        
        Documentation Context: {crag_context}
        
        {user_feedback_context}

        Generate a complete TransformationPlan JSON object with ALL required fields.
        """)
    ]
)

EVALUATOR_CHAIN = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a relevance evaluator for data engineering documentation. Score the relevance of the following document to the given query from 1 to 10, focusing on pandas data cleaning and preprocessing relevance. Provide a brief justification. Format: Score=X\nJustification: [your reasoning]"),
        ("user", "Query: {query}\nDocument: {document}")
    ])
    | llm
)

# Fix the query generator chain to avoid structured output issues
QUERY_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are a data engineering assistant. Based on the provided DataFrame schema's overview summary and user goals,
    generate a concise list of relevant search queries focused on pandas data cleaning and preprocessing techniques.
    Each query should target specific data engineering operations, NOT business logic or calculations.
    
    Return ONLY a JSON array of strings, with each string being a search query.
    Example format: ["Pandas convert object to datetime", "Pandas fillna strategies for numerical data"]
    
    Do not include any other text, just the JSON array.
    """),
    ("user", """
    DataFrame Schema Overview Summary: {data_overview_summary}
    User Overall Goal: {user_instructions}
    
    Generate relevant search queries for data cleaning and preprocessing operations.
    """)
])

def generate_search_queries(data_overview_summary: str, user_instructions: str) -> List[str]:
    """
    Generate search queries using simple LLM call with robust error handling
    """
    try:
        response = llm.invoke(
            QUERY_GENERATOR_PROMPT.format(
                data_overview_summary=data_overview_summary,
                user_instructions=user_instructions if user_instructions else "Clean and standardize data for analysis"
            )
        )
        
        # Extract content safely
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Try multiple parsing strategies
        queries = []
        
        # Strategy 1: Direct JSON parsing
        try:
            import json
            import re
            
            # Find JSON array in response (more robust pattern)
            json_patterns = [
                r'\[.*?\]',  # Standard array
                r'```json\s*(\[.*?\])\s*```',  # JSON code block
                r'```\s*(\[.*?\])\s*```',  # Generic code block
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                    try:
                        queries = json.loads(json_str)
                        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                            print(f"✅ Successfully parsed {len(queries)} queries using JSON")
                            return queries[:10]  # Safe slicing
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"JSON parsing failed: {e}")
        
        # Strategy 2: Line-by-line extraction (FIXED)
        try:
            lines = response_text.split('\n')
            queries = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                    
                # Clean up the line safely
                line = re.sub(r'^["\']|["\']$', '', line)  # Remove quotes
                line = re.sub(r'^[-*•]\s*', '', line)  # Remove list markers
                line = re.sub(r'^\d+\.\s*', '', line)  # Remove numbers
                
                # Validate and add
                if line and len(line) > 5 and len(line) < 100:  # Reasonable length bounds
                    queries.append(line)
                    
                # Safety limit
                if len(queries) >= 10:
                    break
            
            if queries:
                print(f"✅ Successfully extracted {len(queries)} queries using line parsing")
                return queries
                
        except Exception as e:
            print(f"Line parsing failed: {e}")
        
        # Strategy 3: Keyword-based extraction
        try:
            # Look for pandas-related terms in the response
            pandas_keywords = ['pandas', 'dataframe', 'fillna', 'astype', 'groupby', 'merge', 'to_datetime']
            
            sentences = re.split(r'[.!?]', response_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in pandas_keywords):
                    if 5 < len(sentence) < 100:  # Reasonable length
                        queries.append(sentence)
                        if len(queries) >= 6:
                            break
            
            if queries:
                print(f"✅ Successfully extracted {len(queries)} queries using keyword matching")
                return queries
                
        except Exception as e:
            print(f"Keyword extraction failed: {e}")
            
    except Exception as e:
        print(f"Error in query generation: {e}")
    
    # Final fallback with data-specific queries
    print("🔄 Using enhanced fallback queries")
    
    # Generate fallback queries based on data overview
    fallback_queries = [
        "Pandas convert object to datetime",
        "Pandas fillna strategies for numerical data", 
        "Pandas remove duplicates",
        "Pandas handle missing values in categorical columns",
        "Pandas standardize column names",
        "Pandas data type optimization"
    ]
    
    # Add data-specific queries if possible
    if data_overview_summary:
        overview_lower = data_overview_summary.lower()
        
        if 'date' in overview_lower or 'time' in overview_lower:
            fallback_queries.append("Pandas datetime parsing best practices")
            
        if 'object' in overview_lower or 'string' in overview_lower:
            fallback_queries.append("Pandas string data cleaning techniques")
            
        if 'float' in overview_lower or 'numeric' in overview_lower:
            fallback_queries.append("Pandas numerical data preprocessing")
            
        if 'missing' in overview_lower or 'null' in overview_lower:
            fallback_queries.append("Pandas missing data imputation strategies")
    
    return fallback_queries[:8]  # Return reasonable number

# === ENHANCED CRAG FUNCTION ===

def optimize_context_for_llm(data_schema: DataFrameSchema, crag_context: str, max_columns: int = 20, max_context_chars: int = 3000) -> Tuple[str, str]:
    """
    Optimize data schema and CRAG context to fit within LLM context limits
    """
    try:
        # 1. Optimize data schema - summarize instead of full JSON
        schema_summary = {
            "total_columns": len(data_schema.columns) if data_schema.columns else 0,
            "overview": data_schema.overview_summary,
            "sample_columns": []
        }
        
        # Include only most important columns
        if data_schema.columns:
            # Prioritize columns with issues (nulls, mixed types, dates, etc.)
            important_columns = []
            for col in data_schema.columns[:max_columns]:
                col_info = {
                    "name": getattr(col, 'name', str(col)),
                    "dtype": getattr(col, 'dtype', 'unknown'),
                    "nullable": getattr(col, 'is_nullable', True)
                }
                important_columns.append(col_info)
            
            schema_summary["sample_columns"] = important_columns
        
        optimized_schema = json.dumps(schema_summary, indent=2)
        
        # 2. Optimize CRAG context - truncate if too long
        optimized_context = crag_context
        if len(crag_context) > max_context_chars:
            # Keep most relevant parts
            context_parts = crag_context.split("\n\n")
            total_length = 0
            kept_parts = []
            
            for part in context_parts:
                if total_length + len(part) <= max_context_chars:
                    kept_parts.append(part)
                    total_length += len(part)
                else:
                    break
            
            optimized_context = "\n\n".join(kept_parts)
            if len(optimized_context) < len(crag_context):
                optimized_context += "\n\n[Context truncated to fit model limits]"
        
        print(f"📊 Context optimization:")
        print(f"   Original schema length: ~{len(data_schema.model_dump_json())}")
        print(f"   Optimized schema length: {len(optimized_schema)}")
        print(f"   Original context length: {len(crag_context)}")
        print(f"   Optimized context length: {len(optimized_context)}")
        
        return optimized_schema, optimized_context
        
    except Exception as e:
        print(f"Error optimizing context: {e}")
        # Fallback: very minimal context
        minimal_schema = json.dumps({
            "overview": getattr(data_schema, 'overview_summary', 'Dataset with multiple columns'),
            "total_columns": len(data_schema.columns) if hasattr(data_schema, 'columns') and data_schema.columns else 0
        })
        minimal_context = crag_context[:1000] + "... [truncated]" if len(crag_context) > 1000 else crag_context
        return minimal_schema, minimal_context

def enhanced_crag_search_and_grade(queries: List[str], user_instructions: str, data_overview: str) -> Tuple[List[str], bool]:
    """
    Enhanced CRAG implementation with Advanced RAG as primary method and Tavily as fallback.
    """
    
    # === STRATEGY 1: ADVANCED RAG (PRIMARY) ===
    if ADVANCED_RAG_AVAILABLE:
        print("🚀 --- Using Advanced RAG System (Primary) ---")
        
        try:
            # Initialize RAG system
            rag_system = create_rag_system()
            relevant_docs = []
            
            # Process queries with Advanced RAG
            for query in queries[:8]:  # Process first 8 queries for efficiency
                print(f"🔍 Advanced RAG search: {query}")
                
                try:
                    # Get context using semantic search
                    context = rag_system.get_context_for_query(query, max_length=800)
                    
                    if context and context != "No relevant documentation found.":
                        # Format as document for compatibility with existing system
                        formatted_doc = f"Query: {query}\n\nRelevant Documentation:\n{context}"
                        relevant_docs.append(formatted_doc)
                        print(f"✅ Found relevant context for: {query}")
                    else:
                        print(f"❌ No context found for: {query}")
                        
                except Exception as e:
                    print(f"⚠️ RAG search failed for '{query}': {e}")
                    continue
            
            if relevant_docs:
                print(f"✅ Advanced RAG found {len(relevant_docs)} relevant contexts")
                return relevant_docs, True
            else:
                print("⚠️ Advanced RAG found no relevant contexts, falling back to Tavily...")
                
        except Exception as e:
            print(f"❌ Advanced RAG system failed: {e}")
            print("🔄 Falling back to Tavily search...")
    
    else:
        print("⚠️ Advanced RAG not available, using Tavily search")
    
    # === STRATEGY 2: TAVILY + LLM EVALUATION (FALLBACK) ===
    print("🔄 --- Using Tavily + LLM Evaluation (Fallback) ---")
    
    # Detect data characteristics for adaptive search
    data_characteristics = detect_data_characteristics(data_overview)
    print(f"--- Detected data characteristics: {data_characteristics} ---")
    
    # Adjust threshold based on characteristics and query specificity
    dynamic_threshold = adjust_relevance_threshold_dynamically(queries, data_characteristics)
    print(f"--- Using dynamic relevance threshold: {dynamic_threshold} ---")
    
    relevant_docs = []
    
    # Execute main search loop with simplified fallback
    for query in queries[:5]:  # Limit queries to avoid too many API calls
        print(f"--- Tavily Agent Searching: {query} ---")
        try:
            raw_tavily_results = search_tavily(query)
            
            # Handle different Tavily response formats
            if isinstance(raw_tavily_results, str):
                try:
                    raw_tavily_results = json.loads(raw_tavily_results)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON from Tavily for '{query}'. Skipping.")
                    continue
            
            if isinstance(raw_tavily_results, dict):
                raw_tavily_results = [raw_tavily_results]
            
            if not isinstance(raw_tavily_results, list):
                print(f"Warning: Unexpected Tavily response format for '{query}'. Skipping.")
                continue

            # Evaluate each document
            for doc_data in raw_tavily_results:
                doc_content = doc_data.get('content', '')
                if not doc_content:
                    continue

                evaluation_response = EVALUATOR_CHAIN.invoke({
                    "query": query, 
                    "document": doc_content
                })
                
                # Safe content extraction
                try:
                    if hasattr(evaluation_response, 'content'):
                        response_content = evaluation_response.content
                    elif isinstance(evaluation_response, str):
                        response_content = evaluation_response
                    else:
                        response_content = str(evaluation_response)
                    
                    score_match = re.search(r"Score=(\d+)", response_content)
                    score = int(score_match.group(1)) if score_match else 0
                    
                    if score >= dynamic_threshold:
                        relevant_doc = f"Document Title: {doc_data.get('title', 'N/A')}\nContent: {doc_content}"
                        relevant_docs.append(relevant_doc)
                        
                except Exception as eval_error:
                    print(f"  Error evaluating document for '{query}': {eval_error}")
                    # If evaluation fails, include document anyway for better recall
                    relevant_doc = f"Document Title: {doc_data.get('title', 'N/A')}\nContent: {doc_content}"
                    relevant_docs.append(relevant_doc)

        except Exception as e:
            print(f"  Error processing search for '{query}': {e}")
            continue
    
    has_relevant_docs = len(relevant_docs) > 0
    
    if has_relevant_docs:
        print(f"✅ Found {len(relevant_docs)} relevant documents using fallback methods")
    else:
        print("❌ No relevant documents found")
    
    return relevant_docs, has_relevant_docs

# === MAIN TRANSFORMATION PLANNER NODE ===

def print_rag_status():
    """Print RAG system status for user information"""
    if ADVANCED_RAG_AVAILABLE:
        print("🚀 Advanced RAG System: ENABLED")
        print("   - Semantic search with embeddings")
        print("   - Curated pandas documentation")
        print("   - Hybrid retrieval (semantic + keyword)")
        print("   - Fallback to Tavily if needed")
    else:
        print("⚠️ Advanced RAG System: DISABLED")
        print("   - Using Tavily search only")
        print("   - To enable: pip install sentence-transformers faiss-cpu")

def transformation_planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Data Engineering Agent responsible for generating data cleaning and preprocessing plans.
    Implements Enhanced CRAG with Advanced RAG as primary method and fallbacks.
    """
    try:
        print("🔍 DEBUG: Starting transformation_planner_node")
        
        data_schema: DataFrameSchema = state["df_schema"]
        user_instructions: Optional[str] = state.get("user_instructions")
        user_approval: Optional[UserApproval] = state.get("user_approval")

        # Print RAG system status
        print_rag_status()

        # Prepare feedback context for the prompt
        user_feedback_context = ""
        if user_approval and not user_approval.approved and user_approval.feedback:
            user_feedback_context = f"\nUser Feedback for Rejection (INCORPORATE THIS FEEDBACK INTO THE NEW PLAN): {user_approval.feedback}\n"
            print(f"\n--- Incorporating User Feedback: {user_approval.feedback} ---")

        print("🔍 DEBUG: About to start CRAG")
        
        # === ENHANCED DATA-ENGINEERING FOCUSED CRAG ===
        print("\n--- Starting Enhanced Data Engineering CRAG ---")

        # Generate initial queries
        print("--- Dynamically generating data engineering search queries ---")
        try:
            print("🔍 DEBUG: Calling generate_search_queries")
            generated_queries_obj = generate_search_queries(
                data_schema.overview_summary,
                user_instructions or "Clean and standardize data for analysis"
            )
            original_queries = generated_queries_obj
            print(f"Original queries generated: {original_queries}")
            print("🔍 DEBUG: generate_search_queries completed successfully")
            
            # Enhance with data engineering focus
            print("🔍 DEBUG: Calling enhance_queries_with_data_engineering_focus")
            enhanced_queries = enhance_queries_with_data_engineering_focus(
                original_queries, 
                user_instructions or "",
                data_schema.overview_summary
            )
            print(f"Enhanced data engineering queries ({len(enhanced_queries)} total)")
            print("🔍 DEBUG: enhance_queries_with_data_engineering_focus completed")
            
        except Exception as e:
            print(f"🔍 DEBUG: Error in query generation: {e}")
            print(f"🔍 DEBUG: Traceback: {traceback.format_exc()}")
            enhanced_queries = [
                "pandas data cleaning workflow",
                "pandas preprocessing best practices",
                "pandas data type optimization",
                "pandas missing data handling",
                "pandas duplicate removal",
                "pandas column standardization"
            ]

        # Execute enhanced CRAG search with Advanced RAG
        print(f"--- Processing {len(enhanced_queries)} data engineering focused queries ---")
        
        try:
            print("🔍 DEBUG: Calling enhanced_crag_search_and_grade")
            relevant_docs, has_relevant_docs = enhanced_crag_search_and_grade(
                enhanced_queries, 
                user_instructions or "",
                data_schema.overview_summary
            )
            print("🔍 DEBUG: enhanced_crag_search_and_grade completed")
        except Exception as e:
            print(f"🔍 DEBUG: Error in CRAG search: {e}")
            print(f"🔍 DEBUG: Traceback: {traceback.format_exc()}")
            relevant_docs, has_relevant_docs = [], False

        # Prepare context for transformation planning
        if not has_relevant_docs:
            crag_context_str = "No specific relevant context found from RAG. Rely on general pandas data engineering knowledge."
        else:
            crag_context_str = "\n\n".join(relevant_docs)

        print("🔍 DEBUG: About to generate transformation plan")
        
        # === GENERATE DATA CLEANING TRANSFORMATION PLAN ===
        print("\n--- Generating Data Engineering Transformation Plan ---")
        try:
            print("🔍 DEBUG: About to call LLM")
            
            # ✅ ADICIONE ESTAS LINHAS NOVAS:
            # Optimize context to fit model limits
            print("🔍 DEBUG: Optimizing context for model limits")
            optimized_schema, optimized_context = optimize_context_for_llm(data_schema, crag_context_str)
            
            # Use direct LLM call for better control
            response = llm.invoke(
                PLANNER_PROMPT.format(
                    data_schema_json=optimized_schema,  # <- MUDANÇA PRINCIPAL
                    user_instructions=user_instructions if user_instructions else "Clean and standardize data for analysis",
                    crag_context=optimized_context,  # <- MUDANÇA PRINCIPAL
                    user_feedback_context=user_feedback_context
                )
            )
            
            # Extract content safely
            try:
                print("🔍 DEBUG: Extracting response content")
                if hasattr(response, 'content'):
                    response_text = response.content
                elif isinstance(response, str):
                    response_text = response
                else:
                    response_text = str(response)
                print("🔍 DEBUG: Response content extracted successfully")
            except Exception as content_error:
                print(f"🔍 DEBUG: Error extracting response content: {content_error}")
                print(f"🔍 DEBUG: Traceback: {traceback.format_exc()}")
                response_text = str(response)
                
            print(f"Raw LLM response length: {len(response_text)}")
            
            # Try to find and parse JSON
            try:
                print("🔍 DEBUG: About to parse JSON - looking for JSON block")
                # Look for JSON block
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                print(f"🔍 DEBUG: json_start={json_start}, json_end={json_end}")
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    print(f"Extracted JSON length: {len(json_str)}")
                    print("🔍 DEBUG: About to parse JSON string")
                    
                    # Parse JSON
                    plan_dict = json.loads(json_str)
                    print("✅ Successfully parsed JSON")
                    print("🔍 DEBUG: JSON parsing successful")
                    
                    # Validate required fields and fix if needed
                    print("🔍 DEBUG: About to fix transformation plan dict")
                    plan_dict = fix_transformation_plan_dict(plan_dict, data_schema)
                    print("🔍 DEBUG: fix_transformation_plan_dict completed")
                    
                    # Create TransformationPlan from dict
                    print("🔍 DEBUG: About to create TransformationPlan")
                    transformation_plan = TransformationPlan(**plan_dict)
                    print("✅ Successfully created TransformationPlan")
                    
                else:
                    print("🔍 DEBUG: No JSON block found, raising ValueError")
                    raise ValueError("No JSON block found in response")
                    
            except (json.JSONDecodeError, ValueError) as parse_error:
                print(f"❌ JSON parsing failed: {parse_error}")
                print(f"🔍 DEBUG: JSON parsing error traceback: {traceback.format_exc()}")
                print("🔄 Creating manual transformation plan...")
                transformation_plan = create_manual_transformation_plan(data_schema, user_instructions, crag_context_str)

            if transformation_plan:
                print("\n--- Transformation Plan (Pydantic JSON) ---")
                print(transformation_plan.model_dump_json(indent=2))
                print("\n--- Transformation Plan Summary ---")
                print(transformation_plan.overall_summary)
                print("Note that the 'TransformationPlan' includes the following steps:")
                for i, step in enumerate(transformation_plan.transformation_steps):
                    print(f"{i+1}. {step.justification} (Operation Type: {step.operation_type})")
                print(f"The final output will be in {transformation_plan.final_output_format} format.")

                return {"transformation_plan": transformation_plan, "error_message": None}
            else:
                raise ValueError("No transformation plan created")

        except Exception as e:
            print(f"Error generating transformation plan: {e}")
            print(f"🔍 DEBUG: Main error traceback: {traceback.format_exc()}")
            
            # Final fallback
            print("Creating basic fallback transformation plan...")
            try:
                fallback_plan = create_manual_transformation_plan(data_schema, user_instructions, crag_context_str)
                return {"transformation_plan": fallback_plan, "error_message": None}
            except Exception as fallback_error:
                print(f"Fallback plan creation failed: {fallback_error}")
                print(f"🔍 DEBUG: Fallback error traceback: {traceback.format_exc()}")
                return {"transformation_plan": None, "error_message": f"Planner Node Error: {e}"}
                
    except Exception as main_error:
        print(f"🔍 DEBUG: Main function error: {main_error}")
        print(f"🔍 DEBUG: Main function traceback: {traceback.format_exc()}")
        return {"transformation_plan": None, "error_message": f"Main Node Error: {main_error}"}

def fix_transformation_plan_dict(plan_dict: dict, data_schema: DataFrameSchema) -> dict:
    """
    Fix common issues in the transformation plan dictionary
    """
    # Ensure all required fields exist
    if "initial_data_overview_summary" not in plan_dict:
        plan_dict["initial_data_overview_summary"] = data_schema.overview_summary
    
    if "transformation_steps" not in plan_dict:
        plan_dict["transformation_steps"] = []
    
    if "final_output_format" not in plan_dict:
        plan_dict["final_output_format"] = "parquet"
    
    if "overall_summary" not in plan_dict:
        plan_dict["overall_summary"] = "Data cleaning and standardization pipeline"
    
    if "requires_confirmation" not in plan_dict:
        plan_dict["requires_confirmation"] = True
    
    # Fix transformation steps
    fixed_steps = []
    for step in plan_dict.get("transformation_steps", []):
        if isinstance(step, dict):
            # Ensure all required fields exist
            if "justification" not in step:
                step["justification"] = f"Apply {step.get('operation_type', 'transformation')} to {step.get('column_name', 'column')}"
            
            if "expected_outcome" not in step:
                step["expected_outcome"] = f"Improved data quality for {step.get('column_name', 'column')}"
            
            if "params" not in step:
                step["params"] = {}
            
            fixed_steps.append(step)
    
    plan_dict["transformation_steps"] = fixed_steps
    return plan_dict

def create_manual_transformation_plan(data_schema: DataFrameSchema, user_instructions: str, crag_context: str) -> TransformationPlan:

    """
    Create a robust manual transformation plan when LLM fails
    """
    steps = []
    
    try:
        # Safely extract column information
        column_names = []
        columns_info = []
        
        if hasattr(data_schema, 'columns') and data_schema.columns:
            for col in data_schema.columns:
                try:
                    col_name = getattr(col, 'name', str(col))
                    col_dtype = getattr(col, 'dtype', 'unknown')
                    col_nullable = getattr(col, 'is_nullable', True)
                    
                    column_names.append(col_name)
                    columns_info.append({
                        'name': col_name,
                        'dtype': col_dtype,
                        'nullable': col_nullable
                    })
                except Exception as e:
                    print(f"Warning: Error processing column {col}: {e}")
                    continue
        
        print(f"Processing {len(column_names)} columns for manual plan")
        
        # 1. Column standardization (ALWAYS SAFE)
        columns_needing_standardization = []
        for col_name in column_names:
            if ' ' in col_name or '-' in col_name or any(c in col_name for c in ['(', ')', '/', '\\', '.', ':']):
                columns_needing_standardization.append(col_name)
        
        for col_name in columns_needing_standardization[:10]:  # Limit to avoid too many steps
            try:
                steps.append(TransformationStep(
                    operation_type=OperationType.STANDARDIZE_COLUMN_NAME,
                    column_name=col_name,
                    params={},
                    justification=f"Standardize column name '{col_name}' to follow snake_case convention",
                    expected_outcome=f"Column renamed to snake_case format"
                ))
            except Exception as e:
                print(f"Error creating standardization step for {col_name}: {e}")
                continue
        
        # 2. Missing value handling for object/categorical columns
        object_columns = [col for col in columns_info if col['dtype'] == 'object' and col['nullable']]
        for col_info in object_columns[:8]:  # Limit steps
            try:
                steps.append(TransformationStep(
                    operation_type=OperationType.HANDLE_NULLS,
                    column_name=col_info['name'],
                    params={"strategy": "fill_custom", "value": "N/A"},
                    justification=f"Handle missing values in categorical column '{col_info['name']}'",
                    expected_outcome="Missing values filled with 'N/A'"
                ))
            except Exception as e:
                print(f"Error creating null handling step for {col_info['name']}: {e}")
                continue
        
        # 3. Date conversion for date-like columns
        date_columns = [col for col in columns_info if col['dtype'] == 'object' and 
                       any(date_word in col['name'].lower() for date_word in ['data', 'date', 'time', 'created', 'updated'])]
        
        for col_info in date_columns[:3]:  # Limit to 3 date columns
            try:
                steps.append(TransformationStep(
                    operation_type=OperationType.CONVERT_TO_DATETIME,
                    column_name=col_info['name'],
                    params={"format": "auto", "errors": "coerce"},
                    justification=f"Convert '{col_info['name']}' to datetime format for proper temporal analysis",
                    expected_outcome="Column converted to datetime type"
                ))
            except Exception as e:
                print(f"Error creating datetime conversion step for {col_info['name']}: {e}")
                continue
        
        # 4. Numeric conversion for value/price columns
        value_columns = [col for col in columns_info if col['dtype'] == 'object' and 
                        any(value_word in col['name'].lower() for value_word in ['valor', 'price', 'preco', 'cost', 'amount', 'money'])]
        
        for col_info in value_columns[:3]:  # Limit to 3 value columns
            try:
                steps.append(TransformationStep(
                    operation_type=OperationType.CONVERT_TO_NUMERIC,
                    column_name=col_info['name'],
                    params={"decimal_separator": "comma", "remove_currency": True, "errors": "coerce"},
                    justification=f"Convert '{col_info['name']}' to numeric format with Brazilian decimal convention",
                    expected_outcome="Column converted to numeric type with proper decimal format"
                ))
            except Exception as e:
                print(f"Error creating numeric conversion step for {col_info['name']}: {e}")
                continue
        
        # 5. Duplicate removal (always safe if we have multiple columns)
        if len(column_names) > 2:
            try:
                steps.append(TransformationStep(
                    operation_type=OperationType.REMOVE_DUPLICATES,
                    column_name="all_columns",
                    params={"keep": "first"},
                    justification="Remove duplicate rows to ensure data quality",
                    expected_outcome="Duplicate rows removed, keeping first occurrence"
                ))
            except Exception as e:
                print(f"Error creating duplicate removal step: {e}")
        
        # Ensure we have at least some steps
        if not steps:
            print("⚠️ No specific steps created, adding generic cleanup steps")
            try:
                # Add basic steps that should always work
                steps = [
                    TransformationStep(
                        operation_type=OperationType.STANDARDIZE_COLUMN_NAME,
                        column_name="all_columns",
                        params={},
                        justification="Standardize all column names to snake_case format",
                        expected_outcome="Consistent column naming convention"
                    )
                ]
                
                # Add null handling if we have any columns
                if column_names:
                    steps.append(TransformationStep(
                        operation_type=OperationType.HANDLE_NULLS,
                        column_name=column_names[0],  # Use first column as example
                        params={"strategy": "fill_custom", "value": "N/A"},
                        justification=f"Handle missing values in column '{column_names[0]}'",
                        expected_outcome="Missing values properly handled"
                    ))
                    
            except Exception as e:
                print(f"Error creating generic steps: {e}")
                # Ultimate fallback
                steps = [
                    TransformationStep(
                        operation_type=OperationType.STANDARDIZE_COLUMN_NAME,
                        column_name="all_columns",
                        params={},
                        justification="Basic data standardization",
                        expected_outcome="Improved data consistency"
                    )
                ]
        
        # Get overview summary safely
        try:
            overview_summary = getattr(data_schema, 'overview_summary', "Dataset overview not available")
        except Exception:
            overview_summary = "Dataset overview not available"
        
        # Create the transformation plan
        plan = TransformationPlan(
            initial_data_overview_summary=overview_summary,
            transformation_steps=steps[:12],  # Limit to 12 steps for performance
            final_output_format="parquet",
            overall_summary=f"Comprehensive data cleaning pipeline with {len(steps[:12])} transformation steps including column standardization, missing value handling, data type conversions, and duplicate removal.",
            requires_confirmation=True
        )
        
        print(f"✅ Created manual plan with {len(plan.transformation_steps)} steps")
        return plan
        
    except Exception as e:
        print(f"❌ Error in create_manual_transformation_plan: {e}")
        print("🔄 Creating ultra-safe fallback plan")
        
        # Ultra-safe fallback that should never fail
        try:
            return TransformationPlan(
                initial_data_overview_summary="Dataset requires basic cleaning",
                transformation_steps=[
                    TransformationStep(
                        operation_type=OperationType.STANDARDIZE_COLUMN_NAME,
                        column_name="all_columns",
                        params={},
                        justification="Apply basic data standardization to improve consistency",
                        expected_outcome="Standardized dataset ready for analysis"
                    )
                ],
                final_output_format="parquet",
                overall_summary="Basic data cleaning pipeline to ensure data consistency",
                requires_confirmation=True
            )
        except Exception as final_error:
            print(f"❌ Even ultra-safe fallback failed: {final_error}")
            raise Exception(f"Complete failure in manual plan creation: {final_error}")