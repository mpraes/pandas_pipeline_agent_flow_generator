# src/agents/transformation_planner_agent.py
# VERSÃO CORRIGIDA COM TODOS OS IMPORTS E DEFINIÇÕES

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

from src.core.data_schema import GraphState, TransformationPlan, DataFrameSchema, TransformationStep, OperationType, SearchQueries, UserApproval
from src.core.llm_config import get_llm
from src.core.tavily_utils import search_tavily

# IMPORT PARA RAG AVANÇADO
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

# === TODAS AS FUNÇÕES AUXILIARES NECESSÁRIAS ===

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

# === PROMPTS E CHAINS ===

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", f"""
        You are an expert Data Engineering Agent specializing in Pandas data transformations.
        
        **CRITICAL**: You MUST generate a valid JSON object that matches the TransformationPlan schema EXACTLY.
        
        Required fields for each transformation step:
        - operation_type: Must be one of these EXACT values: {", ".join([f"'{op.value}'" for op in OperationType])}
        - column_name: Exact column name from the schema
        - params: Dictionary (can be empty {{}})
        - justification: Clear explanation of why this step is needed
        - expected_outcome: What the result will be
        
        Focus EXCLUSIVELY on data engineering and cleaning operations. NO business logic.
        
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

QUERY_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are a data engineering assistant. Based on the provided DataFrame schema's overview summary and user goals,
    generate a concise list of relevant search queries focused on pandas data cleaning and preprocessing techniques.
    Each query should target specific data engineering operations, NOT business logic or calculations.
    Return only a JSON list of strings, with each string being a search query.
    Example: ["Pandas convert object to datetime", "Pandas fillna strategies for numerical data"]
    """),
    ("user", """
    DataFrame Schema Overview Summary: {data_overview_summary}
    User Overall Goal: {user_instructions}
    
    Generate relevant search queries for data cleaning and preprocessing operations.
    """)
])

query_generator_chain = QUERY_GENERATOR_PROMPT | llm.with_structured_output(SearchQueries)

# === ENHANCED CRAG FUNCTION ===

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
                
                score_match = re.search(r"Score=(\d+)", evaluation_response.content)
                score = int(score_match.group(1)) if score_match else 0
                
                if score >= dynamic_threshold:
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

    # === ENHANCED DATA-ENGINEERING FOCUSED CRAG ===
    print("\n--- Starting Enhanced Data Engineering CRAG ---")

    # Generate initial queries
    print("--- Dynamically generating data engineering search queries ---")
    try:
        generated_queries_obj = query_generator_chain.invoke({
            "data_overview_summary": data_schema.overview_summary,
            "user_instructions": user_instructions if user_instructions else "Clean and standardize data for analysis"
        })
        original_queries = generated_queries_obj.queries
        print(f"Original queries generated: {original_queries}")
        
        # Enhance with data engineering focus
        enhanced_queries = enhance_queries_with_data_engineering_focus(
            original_queries, 
            user_instructions or "",
            data_schema.overview_summary
        )
        print(f"Enhanced data engineering queries ({len(enhanced_queries)} total)")
        
    except Exception as e:
        print(f"Error generating search queries: {e}. Using default data engineering queries.")
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
    
    relevant_docs, has_relevant_docs = enhanced_crag_search_and_grade(
        enhanced_queries, 
        user_instructions or "",
        data_schema.overview_summary
    )

    # Prepare context for transformation planning
    if not has_relevant_docs:
        crag_context_str = "No specific relevant context found from RAG. Rely on general pandas data engineering knowledge."
    else:
        crag_context_str = "\n\n".join(relevant_docs)

    # === GENERATE DATA CLEANING TRANSFORMATION PLAN ===
    print("\n--- Generating Data Engineering Transformation Plan ---")
    try:
        # Use direct LLM call for better control
        response = llm.invoke(
            PLANNER_PROMPT.format(
                data_schema_json=data_schema.model_dump_json(indent=2),
                user_instructions=user_instructions if user_instructions else "Clean and standardize data for analysis",
                crag_context=crag_context_str,
                user_feedback_context=user_feedback_context
            )
        )
        
        # Extract content
        response_text = response.content if hasattr(response, 'content') else str(response)
        print(f"Raw LLM response length: {len(response_text)}")
        
        # Try to find and parse JSON
        try:
            # Look for JSON block
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                print(f"Extracted JSON length: {len(json_str)}")
                
                # Parse JSON
                plan_dict = json.loads(json_str)
                print("✅ Successfully parsed JSON")
                
                # Validate required fields and fix if needed
                plan_dict = fix_transformation_plan_dict(plan_dict, data_schema)
                
                # Create TransformationPlan from dict
                transformation_plan = TransformationPlan(**plan_dict)
                print("✅ Successfully created TransformationPlan")
                
            else:
                raise ValueError("No JSON block found in response")
                
        except (json.JSONDecodeError, ValueError) as parse_error:
            print(f"❌ JSON parsing failed: {parse_error}")
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
        
        # Final fallback
        print("Creating basic fallback transformation plan...")
        try:
            fallback_plan = create_manual_transformation_plan(data_schema, user_instructions, crag_context_str)
            return {"transformation_plan": fallback_plan, "error_message": None}
        except Exception as fallback_error:
            print(f"Fallback plan creation failed: {fallback_error}")
            return {"transformation_plan": None, "error_message": f"Planner Node Error: {e}"}

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
    Create a manual transformation plan when LLM fails
    """
    steps = []
    
    # Get actual column names from schema
    if hasattr(data_schema, 'columns') and data_schema.columns:
        column_names = [col.name for col in data_schema.columns]
        
        # Add column standardization steps for columns with spaces/special chars
        for col_name in column_names:
            if ' ' in col_name or '-' in col_name or any(c in col_name for c in ['(', ')', '/', '\\']):
                steps.append(TransformationStep(
                    operation_type=OperationType.STANDARDIZE_COLUMN_NAME,
                    column_name=col_name,
                    params={},
                    justification=f"Standardize column name '{col_name}' to follow snake_case convention",
                    expected_outcome=f"Column renamed to snake_case format"
                ))
        
        # Add missing value handling for columns with nulls
        for col in data_schema.columns:
            if col.is_nullable and col.dtype == "object":
                steps.append(TransformationStep(
                    operation_type=OperationType.HANDLE_NULLS,
                    column_name=col.name,
                    params={"strategy": "fill_custom", "value": "N/A"},
                    justification=f"Handle missing values in categorical column '{col.name}'",
                    expected_outcome="Missing values filled with 'N/A'"
                ))
        
        # Add date conversion for columns that might be dates
        for col in data_schema.columns:
            if col.dtype == "object" and any(date_word in col.name.lower() for date_word in ['data', 'date', 'time']):
                steps.append(TransformationStep(
                    operation_type=OperationType.CONVERT_TO_DATETIME,
                    column_name=col.name,
                    params={"format": "auto"},
                    justification=f"Convert '{col.name}' to datetime format for proper temporal analysis",
                    expected_outcome="Column converted to datetime type"
                ))
        
        # Add numeric conversion for value columns  
        for col in data_schema.columns:
            if col.dtype == "object" and any(value_word in col.name.lower() for value_word in ['valor', 'price', 'preco', 'cost']):
                steps.append(TransformationStep(
                    operation_type=OperationType.CONVERT_TO_NUMERIC,
                    column_name=col.name,
                    params={"decimal_separator": "comma", "remove_currency": True},
                    justification=f"Convert '{col.name}' to numeric format with Brazilian decimal convention",
                    expected_outcome="Column converted to numeric type with proper decimal format"
                ))
        
        # Add duplicate removal
        if len(column_names) > 5:  # Only for datasets with multiple columns
            steps.append(TransformationStep(
                operation_type=OperationType.REMOVE_DUPLICATES,
                column_name="all_columns",
                params={},
                justification="Remove duplicate rows to ensure data quality",
                expected_outcome="Duplicate rows removed"
            ))
    
    # If no specific steps, add generic ones
    if not steps:
        steps = [
            TransformationStep(
                operation_type=OperationType.STANDARDIZE_COLUMN_NAME,
                column_name="all_columns",
                params={},
                justification="Standardize all column names to snake_case format",
                expected_outcome="Consistent column naming"
            ),
            TransformationStep(
                operation_type=OperationType.HANDLE_NULLS,
                column_name="all_columns", 
                params={},
                justification="Handle missing values across all columns",
                expected_outcome="Reduced missing data"
            )
        ]
    
    return TransformationPlan(
        initial_data_overview_summary=data_schema.overview_summary,
        transformation_steps=steps[:12],  # Limit to 12 steps
        final_output_format="parquet",
        overall_summary=f"Comprehensive data cleaning pipeline with {len(steps)} transformation steps including column standardization, missing value handling, data type conversions, and duplicate removal.",
        requires_confirmation=True
    )