"""
test_agent.py - Corrected Test Agent (no infinite recursion)
Functional system for automated test generation
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, NamedTuple
from pathlib import Path
import json
from datetime import datetime
import tempfile
import ast
import importlib.util

# LangChain imports
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Functional types
TestResult = NamedTuple('TestResult', [
    ('test_name', str),
    ('test_code', str),
    ('test_type', str),
    ('dependencies', List[str]),
    ('expected_coverage', str),
    ('description', str)
])

PipelineAnalysis = NamedTuple('PipelineAnalysis', [
    ('file_path', str),
    ('code_content', str),
    ('imports', List[str]),
    ('functions', List[str]),
    ('data_operations', List[str]),
    ('complexity_score', int),
    ('has_main', bool),
    ('code_lines', int)
])

# Model configuration
def setup_llm() -> ChatGroq:
    """Configure the Groq model"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not configured")
    
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-70b-versatile",
        temperature=0.1,
        max_tokens=3000
    )

# Code analysis - NO RECURSION
def analyze_pipeline_code(file_path: str) -> PipelineAnalysis:
    """Analyze pipeline code ONCE ONLY"""
    logger.info(f"Analyzing pipeline: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        # Basic analysis with AST
        try:
            tree = ast.parse(code_content)
            imports = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                    
        except SyntaxError:
            logger.warning(f"Syntax error in {file_path}")
            imports = []
            functions = []
        
        # Data operations
        data_operations = []
        pandas_ops = ['read_csv', 'read_parquet', 'to_csv', 'to_parquet', 'dropna', 'fillna', 'merge', 'groupby']
        for op in pandas_ops:
            if op in code_content:
                data_operations.append(op)
        
        # Simple complexity score
        complexity_score = (
            len(functions) * 5 +
            len(data_operations) * 3 +
            code_content.count('if ') * 2 +
            code_content.count('for ') * 2 +
            len(imports)
        )
        
        has_main = 'def main(' in code_content
        code_lines = len(code_content.split('\n'))
        
        return PipelineAnalysis(
            file_path=file_path,
            code_content=code_content[:2000],  # Limit size for LLM
            imports=imports,
            functions=functions,
            data_operations=data_operations,
            complexity_score=complexity_score,
            has_main=has_main,
            code_lines=code_lines
        )
        
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return PipelineAnalysis(
            file_path=file_path,
            code_content="",
            imports=[],
            functions=[],
            data_operations=[],
            complexity_score=0,
            has_main=False,
            code_lines=0
        )

# Test generation with LLM
def generate_tests_with_llm(analysis: PipelineAnalysis) -> List[TestResult]:
    """Generate tests using LLM - SINGLE CALL ONLY"""
    try:
        llm = setup_llm()
        
        prompt = f"""
You are a testing expert for data pipelines. Analyze this Python code and generate 2-3 simple tests.

ANALYZED CODE:
- File: {Path(analysis.file_path).name}
- Functions: {analysis.functions}
- Operations: {analysis.data_operations}
- Has main(): {analysis.has_main}
- Lines: {analysis.code_lines}

CODE SAMPLE:
```python
{analysis.code_content}
```

Generate EXACTLY 2 tests in Python using pytest:
1. A basic import/execution test
2. A data validation test

JSON Format:
{{
  "tests": [
    {{
      "name": "test_basic_import",
      "code": "import pytest\\ndef test_basic_import():\\n    # Test here",
      "type": "unit",
      "description": "Basic test"
    }},
    {{
      "name": "test_data_validation", 
      "code": "def test_data_validation():\\n    # Validation here",
      "type": "integration",
      "description": "Data test"
    }}
  ]
}}

RESPOND ONLY WITH JSON:
"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # Extract JSON
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
                
                tests = []
                for test_info in data.get('tests', [])[:3]:  # Maximum 3 tests
                    tests.append(TestResult(
                        test_name=test_info.get('name', 'test_unnamed'),
                        test_code=test_info.get('code', '# Empty test'),
                        test_type=test_info.get('type', 'unit'),
                        dependencies=['pytest', 'pandas'],
                        expected_coverage=test_info.get('description', ''),
                        description=test_info.get('description', 'Automatically generated test')
                    ))
                
                logger.info(f"Generated {len(tests)} tests via LLM")
                return tests
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
        
    except Exception as e:
        logger.error(f"Error in LLM generation: {e}")
    
    # Fallback: generate basic test
    return [create_fallback_test(analysis)]

def create_fallback_test(analysis: PipelineAnalysis) -> TestResult:
    """Creates basic test when LLM fails"""
    pipeline_name = Path(analysis.file_path).stem.replace('-', '_')
    
    test_code = f"""
import pytest
import pandas as pd
from pathlib import Path
import sys
import os

def test_{pipeline_name}_basic():
    \"\"\"Basic test for pipeline {pipeline_name}\"\"\"
    pipeline_path = "{analysis.file_path}"
    
    # Check if file exists
    assert Path(pipeline_path).exists(), "Pipeline not found"
    
    # Check if it has content
    with open(pipeline_path, 'r') as f:
        content = f.read()
    
    assert len(content) > 100, "Pipeline too small"
    assert 'import' in content, "Pipeline without imports"
    
    print("✅ Basic test passed")

def test_{pipeline_name}_syntax():
    \"\"\"Tests if the code has valid syntax\"\"\"
    with open("{analysis.file_path}", 'r') as f:
        code = f.read()
    
    # Try to compile
    try:
        compile(code, "{analysis.file_path}", 'exec')
        print("✅ Valid syntax")
    except SyntaxError as e:
        pytest.fail(f"Syntax error: {{e}}")
"""
    
    return TestResult(
        test_name=f"test_{pipeline_name}_fallback",
        test_code=test_code,
        test_type="unit",
        dependencies=["pytest", "pandas"],
        expected_coverage="Basic functionality test",
        description="Automatically generated test as fallback"
    )

# Main function NO RECURSION
def run_test_generation_agent(pipeline_path: str) -> Dict[str, Any]:
    """Execute test generation ONCE ONLY"""
    
    if not Path(pipeline_path).exists():
        logger.error(f"Pipeline not found: {pipeline_path}")
        return {"error": "Pipeline not found"}
    
    # 1. Analyze code (once)
    analysis = analyze_pipeline_code(pipeline_path)
    
    # 2. Generate tests (once)
    generated_tests = generate_tests_with_llm(analysis)
    
    # 3. Validate generated tests
    valid_tests = 0
    for test in generated_tests:
        try:
            compile(test.test_code, f"<test_{test.test_name}>", 'exec')
            valid_tests += 1
        except SyntaxError:
            logger.warning(f"Test {test.test_name} has syntax error")
    
    # 4. Build result
    result = {
        "pipeline_path": pipeline_path,
        "analysis": {
            "complexity_score": analysis.complexity_score,
            "functions_found": len(analysis.functions),
            "data_operations": len(analysis.data_operations),
            "has_main": analysis.has_main,
            "code_lines": analysis.code_lines
        },
        "generated_tests": [t._asdict() for t in generated_tests],
        "final_report": {
            "test_generation_summary": {
                "tests_generated": len(generated_tests),
                "syntax_valid_tests": valid_tests
            },
            "analysis_summary": {
                "complexity_score": analysis.complexity_score,
                "functions_found": len(analysis.functions),
                "data_operations": len(analysis.data_operations)
            },
            "recommendations": generate_recommendations(analysis, generated_tests)
        }
    }
    
    logger.info(f"Generated tests: {len(generated_tests)}, Valid: {valid_tests}")
    return result

def generate_recommendations(analysis: PipelineAnalysis, tests: List[TestResult]) -> List[str]:
    """Generates simple recommendations"""
    recommendations = []
    
    if analysis.complexity_score < 10:
        recommendations.append("Pipeline too simple - add more transformations")
    
    if not analysis.has_main:
        recommendations.append("Add main() function for execution")
    
    if len(analysis.data_operations) < 2:
        recommendations.append("Add more data operations")
    
    if len(tests) < 2:
        recommendations.append("Generate more tests for better coverage")
    
    return recommendations

# Functions for integration with main.py
def save_generated_tests(result: Dict[str, Any], pipeline_file: Path, stage: str = "generated"):
    """Saves generated tests to files"""
    generated_tests = result.get('generated_tests', [])
    
    if not generated_tests:
        logger.warning("No tests generated to save")
        return
    
    # Create directory
    tests_dir = Path(f"tests/{stage}")
    tests_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline_name = pipeline_file.stem.replace('-', '_')
    
    for i, test_data in enumerate(generated_tests):
        test_filename = f"test_{pipeline_name}_{i+1}.py"
        test_path = tests_dir / test_filename
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(f'"""\n{test_data.get("description", "Test generated automatically")}\n')
            f.write(f'Stage: {stage}\n')
            f.write(f'Generated on: {datetime.now()}\n"""\n\n')
            f.write(test_data.get("test_code", "# Empty test"))
        
        logger.info(f"Test {stage} saved: {test_path}")

def test_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for integration with main.py"""
    pipeline_path = state.get('current_pipeline_path')
    
    if not pipeline_path:
        # Try to find the most recent pipeline
        pipelines_dir = Path("pipelines/generated")
        if pipelines_dir.exists():
            pipeline_files = list(pipelines_dir.glob("pipeline_*.py"))
            if pipeline_files:
                pipeline_path = str(max(pipeline_files, key=lambda x: x.stat().st_mtime))
    
    if not pipeline_path:
        logger.error("No pipeline found for testing")
        return {**state, 'test_generation_results': None}
    
    # Execute ONCE
    try:
        result = run_test_generation_agent(pipeline_path)
        logger.info("Tests generated successfully via LangGraph node")
        return {**state, 'test_generation_results': result}
    except Exception as e:
        logger.error(f"Error generating tests via node: {e}")
        return {**state, 'test_generation_results': None}

def validate_test_results(test_results: Optional[Dict[str, Any]]) -> tuple[bool, Dict[str, Any]]:
    """Validates if tests should regenerate pipeline"""
    if not test_results:
        return True, {"reason": "Test generation failed", "score": 0}
    
    final_report = test_results.get("final_report", {})
    test_summary = final_report.get("test_generation_summary", {})
    
    tests_generated = test_summary.get("tests_generated", 0)
    syntax_valid_tests = test_summary.get("syntax_valid_tests", 0)
    
    success_rate = syntax_valid_tests / max(tests_generated, 1)
    
    should_regenerate = success_rate < 0.5 or tests_generated < 1
    
    return should_regenerate, {
        "success_rate": success_rate,
        "tests_generated": tests_generated,
        "syntax_valid_tests": syntax_valid_tests
    }

if __name__ == "__main__":
    # Direct test
    import sys
    if len(sys.argv) > 1:
        pipeline_path = sys.argv[1]
        result = run_test_generation_agent(pipeline_path)
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Usage: python test_agent.py <pipeline_path>")