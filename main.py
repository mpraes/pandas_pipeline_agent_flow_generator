# main.py - Corrected version with integrated automated testing

from dotenv import load_dotenv
import os
import sys
from typing import Optional, List

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key or groq_api_key == "your_groq_api_key_here":
    print("ERROR: GROQ_API_KEY is required but not set or is still a placeholder.")
    print("Please ensure your .env file in the project root contains: GROQ_API_KEY=your_actual_api_key_here")
    print("You can get a free API key from: https://console.groq.com/")
    sys.exit(1)

from langgraph.graph import StateGraph, END
from src.core.data_schema import GraphState, DataFrameSchema, TransformationPlan, UserApproval
from src.agents.data_ingestion_agent import data_ingestion_node
from src.agents.transformation_planner_agent import transformation_planner_node
from src.agents.approval_agent import approval_agent_node
from src.agents.code_generation_agent import code_generation_node
from src.agents.code_quality_agent import code_quality_node
from src.agents.refactoring_pipeline_agent import refactoring_pipeline_node, final_quality_check_node

# NEW IMPORT - Test agent (with error handling)
try:
    from src.agents.test_agent import test_generation_node, validate_test_results, run_test_generation_agent, save_generated_tests
    TEST_AGENT_AVAILABLE = True
except ImportError:
    print("⚠️  test_agent.py not found. Automated testing functionality disabled.")
    TEST_AGENT_AVAILABLE = False

from src.utils.data_quality_checklist import (
    get_enhanced_instructions,
    DataQualityCategory
)

# NEW ROUTING FUNCTION - Decision logic based on test results
def route_after_testing(state: GraphState) -> str:
    """
    Determines next step based on generated test results.
    If tests fail too much, goes back to code_generation.
    If they pass, continues to refactoring.
    """
    test_results = state.get("test_generation_results")
    
    # If test_agent is not available, continue the flow
    if not TEST_AGENT_AVAILABLE or not test_results:
        return "continue_pipeline"
    
    # If it was skipped due to unavailability
    if test_results.get("skipped"):
        return "continue_pipeline"
    
    final_report = test_results.get("final_report", {})
    test_summary = final_report.get("test_generation_summary", {})
    
    # Criteria to decide if the pipeline is good
    tests_generated = test_summary.get("tests_generated", 0)
    syntax_valid_tests = test_summary.get("syntax_valid_tests", 0)
    
    # Calculate test success rate
    if tests_generated == 0:
        success_rate = 0
    else:
        success_rate = syntax_valid_tests / tests_generated
    
    # Check complexity and quality of the analyzed pipeline
    analysis_summary = final_report.get("analysis_summary", {})
    complexity_score = analysis_summary.get("complexity_score", 0)
    functions_found = analysis_summary.get("functions_found", 0)
    data_operations = analysis_summary.get("data_operations", 0)
    
    # Decision logic
    # Criteria for code regeneration:
    should_regenerate = (
        success_rate < 0.7 or  # Less than 70% of valid tests
        tests_generated < 2 or  # Too few tests generated
        complexity_score < 10 or  # Pipeline too simple (probably incomplete)
        functions_found == 0 or  # No functions defined
        data_operations < 2  # Insufficient data operations
    )
    
    if should_regenerate:
        # Add detailed feedback for regeneration
        feedback_details = []
        
        if success_rate < 0.7:
            feedback_details.append(f"Low test success rate: {success_rate:.1%}")
        if tests_generated < 2:
            feedback_details.append(f"Too few tests generated: {tests_generated}")
        if complexity_score < 10:
            feedback_details.append("Pipeline too simple, missing operations")
        if functions_found == 0:
            feedback_details.append("No functions defined in pipeline")
        if data_operations < 2:
            feedback_details.append("Insufficient data operations identified")
        
        # Store feedback in state for code_generation_agent to use
        state["regeneration_feedback"] = {
            "reason": "Automated tests failed",
            "details": feedback_details,
            "test_results": test_results,
            "recommendations": final_report.get("recommendations", [])
        }
        
        return "regenerate_code"
    else:
        # Testes passaram, continuar para refactoring
        return "continue_pipeline"

def route_on_approval(state: GraphState) -> str:
    """
    Determines next step based on user approval.
    """
    if state.get("user_approval") and state["user_approval"].approved:
        return "approved"
    else:
        return "rejected"

def route_after_final_testing(state: GraphState) -> str:
    """
    Determines next step based on refactored pipeline tests.
    If tests still fail, can go back to refactoring once.
    """
    test_results = state.get("test_generation_results")
    
    # If test_agent is not available, finish
    if not TEST_AGENT_AVAILABLE or not test_results:
        return "complete_pipeline"
    
    # If it was skipped due to unavailability
    if test_results.get("skipped"):
        return "complete_pipeline"
    
    final_report = test_results.get("final_report", {})
    test_summary = final_report.get("test_generation_summary", {})
    
    # More relaxed criteria for refactored pipeline
    tests_generated = test_summary.get("tests_generated", 0)
    syntax_valid_tests = test_summary.get("syntax_valid_tests", 0)
    
    if tests_generated == 0:
        success_rate = 0
    else:
        success_rate = syntax_valid_tests / tests_generated
    
    # Check if already tried to refactor before
    refactoring_attempts = state.get("refactoring_attempts", 0)
    
    # Criteria for new refactoring (more restrictive)
    should_refactor_again = (
        success_rate < 0.5 and  # Less than 50% of valid tests
        refactoring_attempts < 2 and  # Maximum 2 refactoring attempts
        tests_generated >= 1  # At least 1 test was generated
    )
    
    if should_refactor_again:
        # Increment attempt counter
        state["refactoring_attempts"] = refactoring_attempts + 1
        
        # Add specific feedback for refactoring
        state["refactoring_feedback"] = {
            "reason": "Final tests failed after refactoring",
            "success_rate": success_rate,
            "attempt": refactoring_attempts + 1,
            "recommendations": final_report.get("recommendations", [])
        }
        
        return "regenerate_refactored"
    else:
        # Finalizar pipeline
        return "complete_pipeline"

# NEW FUNCTION - Integrated test generation node
def integrated_test_generation_node(state: GraphState) -> GraphState:
    """
    Integrated node that executes automated test generation
    and analyzes results for flow decision
    """
    if not TEST_AGENT_AVAILABLE:
        print("⚠️  Test agent not available, skipping automated tests")
        return {**state, "test_generation_results": {"skipped": True}}
    
    from pathlib import Path
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Determine which directory to use based on pipeline stage
    is_refactored_stage = state.get("refactored_code_result") is not None
    current_stage = "refactored" if is_refactored_stage else "generated"
    
    # Look for latest pipeline in appropriate directory
    pipelines_dir = Path(f"pipelines/{current_stage}")
    
    # 🔧 FIX: Create directories if they don't exist
    if not pipelines_dir.exists():
        pipelines_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Directory created: {pipelines_dir}")
    
    # If it's refactored but no files, look in generated folder
    pipeline_files = list(pipelines_dir.glob("pipeline_*.py"))
    if not pipeline_files and current_stage == "refactored":
        logger.info("🔄 No refactored pipeline found, using generated")
        pipelines_dir = Path("pipelines/generated")
        pipeline_files = list(pipelines_dir.glob("pipeline_*.py"))
        current_stage = "generated"  # Adjust stage
    
    if not pipeline_files:
        logger.error(f"No pipeline found to test in {current_stage}")
        return {**state, "test_generation_results": None}
    
    # Get the latest one
    latest_pipeline = max(pipeline_files, key=lambda x: x.stat().st_mtime)
    
    logger.info(f"Running automated tests for {current_stage}: {latest_pipeline}")
    
    try:
        # Execute test generation agent
        test_results = run_test_generation_agent(str(latest_pipeline))
        
        # Save generated tests in appropriate directory
        save_generated_tests(test_results, latest_pipeline, stage=current_stage)
        
        # Add stage information to result
        test_results["pipeline_stage"] = current_stage
        test_results["tested_pipeline_path"] = str(latest_pipeline)
        
        logger.info(f"Tests generated and analyzed successfully for {current_stage}")
        return {**state, "test_generation_results": test_results}
        
    except Exception as e:
        logger.error(f"Error in test generation for {current_stage}: {e}")
        return {**state, "test_generation_results": None}

# MODIFIED MAIN FUNCTION
def run_agent_flow(file_path: str, 
                  user_instructions: Optional[str] = None,
                  focus_categories: Optional[List[DataQualityCategory]] = None,
                  min_priority: str = "medium",
                  use_checklist: bool = True,
                  interactive_preferences: bool = True,
                  max_regeneration_attempts: int = 3):
    """
    Executes the agent flow with integrated automated testing and intelligent regeneration
    
    Args:
        file_path: Path to the file to be processed
        user_instructions: Specific user instructions
        focus_categories: Specific checklist categories to focus on
        min_priority: Minimum priority of checklist items
        use_checklist: Whether to use the structured checklist system
        interactive_preferences: Whether to collect preferences interactively
        max_regeneration_attempts: Maximum number of code regeneration attempts
    """
    print(f"Processing file: {file_path}")
    
    # Existing instruction logic...
    if use_checklist:
        enhanced_instructions = get_enhanced_instructions(
            user_instructions=user_instructions,
            focus_categories=focus_categories,
            min_priority=min_priority,
            use_interactive_preferences=interactive_preferences
        )
        instructions_to_use = enhanced_instructions
        if interactive_preferences:
            print("Using enhanced instructions with quality checklist and personalized preferences")
        else:
            print("Using enhanced instructions with quality checklist")
    else:
        instructions_to_use = user_instructions
        print(f"User's overall goal: {user_instructions if user_instructions else 'No specific instructions, infer cleaning needs.'}")
    
    print("\n--- Starting Agent Flow with Automated Testing ---")
    
    # Initialize the graph
    workflow = StateGraph(GraphState)
    
    # Add all existing nodes
    workflow.add_node("data_ingestion", data_ingestion_node)
    workflow.add_node("transformation_planner", transformation_planner_node)
    workflow.add_node("approval", approval_agent_node)
    workflow.add_node("code_generation", code_generation_node)
    workflow.add_node("code_quality", code_quality_node)
    # FIRST TEST - After initial generation
    workflow.add_node("initial_testing", integrated_test_generation_node)
    workflow.add_node("refactoring", refactoring_pipeline_node)
    # SECOND TEST - After refactoring
    workflow.add_node("final_testing", integrated_test_generation_node)
    workflow.add_node("final_quality_check", final_quality_check_node)
    
    # Existing edges
    workflow.add_edge("data_ingestion", "transformation_planner")
    workflow.add_edge("transformation_planner", "approval")
    
    # Conditional approval edge
    workflow.add_conditional_edges(
        "approval",
        route_on_approval,
        {
            "approved": "code_generation",
            "rejected": "transformation_planner"
        }
    )
    
    # Edge from code_generation to code_quality
    workflow.add_edge("code_generation", "code_quality")
    
    # FIRST ROUND OF TESTS - After code_quality (pipelines/generated)
    workflow.add_edge("code_quality", "initial_testing")
    
    # CONDITIONAL EDGE - Based on initial test results
    workflow.add_conditional_edges(
        "initial_testing",
        route_after_testing,
        {
            "regenerate_code": "code_generation",  # Go back to regenerate code
            "continue_pipeline": "refactoring"     # Continue to refactoring
        }
    )
    
    # After refactoring, second round of tests
    workflow.add_edge("refactoring", "final_testing")
    
    # FINAL CONDITIONAL EDGE - Based on refactored pipeline tests
    workflow.add_conditional_edges(
        "final_testing",
        route_after_final_testing,
        {
            "regenerate_refactored": "refactoring",  # Go back to refactoring
            "complete_pipeline": "final_quality_check"  # Finish
        }
    )
    
    # Final edge
    workflow.add_edge("final_quality_check", END)
    
    # Set entry point
    workflow.set_entry_point("data_ingestion")
    
    # Compile the graph
    app = workflow.compile()
    
    # Execute workflow with regeneration control
    initial_state = {
        "file_path": file_path,
        "user_instructions": instructions_to_use,
        "regeneration_attempts": 0,
        "max_regeneration_attempts": max_regeneration_attempts
    }
    
    result = app.invoke(initial_state)
    
    return result

# Updated example functions
def run_with_basic_checklist_and_testing():
    """Runs with basic checklist and automated testing"""
    return run_agent_flow(
        file_path="data/input_samples/taxigov-corridas-completo.csv",
        user_instructions="Clean and standardize the data for analysis",
        use_checklist=True,
        interactive_preferences=False,
        max_regeneration_attempts=2  # Maximum 2 regeneration attempts
    )

def run_with_focused_checklist_and_testing():
    """Runs with focused checklist and automated testing"""
    return run_agent_flow(
        file_path="data/input_samples/taxigov-corridas-completo.csv",
        user_instructions="Clean and standardize the data for analysis",
        focus_categories=[DataQualityCategory.DATA_TYPES, DataQualityCategory.MISSING_VALUES],
        min_priority="high",
        use_checklist=True,
        interactive_preferences=True,
        max_regeneration_attempts=3
    )

def run_without_checklist_but_with_testing():
    """Runs without checklist but with automated testing"""
    return run_agent_flow(
        file_path="data/input_samples/taxigov-corridas-completo.csv",
        user_instructions="Clean and standardize the data for analysis",
        use_checklist=False,
        interactive_preferences=False,
        max_regeneration_attempts=2
    )

if __name__ == "__main__":
    # Usage example with automated testing
    print("=== EXECUTING PIPELINE WITH AUTOMATED TESTING ===")
    result = run_with_basic_checklist_and_testing()
    print("Workflow with automated testing completed successfully!")
    
    # Print test results if available
    if result.get("test_generation_results"):
        test_summary = result["test_generation_results"].get("final_report", {}).get("test_generation_summary", {})
        print(f"\n📊 AUTOMATED TESTING SUMMARY:")
        print(f"   Tests generated: {test_summary.get('tests_generated', 0)}")
        print(f"   Valid tests: {test_summary.get('syntax_valid_tests', 0)}")
        print(f"   Success rate: {test_summary.get('syntax_valid_tests', 0)/max(test_summary.get('tests_generated', 1), 1):.1%}")
    
    if result.get("regeneration_feedback"):
        print(f"\n🔄 REGENERATIONS PERFORMED:")
        feedback = result["regeneration_feedback"]
        print(f"   Reason: {feedback['reason']}")
        for detail in feedback.get('details', []):
            print(f"   - {detail}")