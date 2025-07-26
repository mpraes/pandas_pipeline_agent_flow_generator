# main.py
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
from src.agents.code_generation_agent import code_generation_node  # NEW NODE
from src.utils.data_quality_checklist import (
    get_enhanced_instructions,
    DataQualityCategory
)

# Define the conditional logic for the graph
def route_on_approval(state: GraphState) -> str:
    """
    Determines the next step based on user approval.
    """
    if state.get("user_approval") and state["user_approval"].approved:
        return "approved"
    else:
        # If rejected, loop back to the transformation planner with feedback
        return "rejected"

# --- Main Application Logic ---
def run_agent_flow(file_path: str, 
                  user_instructions: Optional[str] = None,
                  focus_categories: Optional[List[DataQualityCategory]] = None,
                  min_priority: str = "medium",
                  use_checklist: bool = True,
                  interactive_preferences: bool = True):
    """
    Executes the agent flow with option to use structured checklist and interactive preferences
    
    Args:
        file_path: Path to the file to be processed
        user_instructions: Specific user instructions
        focus_categories: Specific checklist categories to focus on (optional)
        min_priority: Minimum priority of checklist items ("high", "medium", "low")
        use_checklist: Whether to use the structured checklist system
        interactive_preferences: Whether to collect preferences interactively
    """
    print(f"Processing file: {file_path}")
    
    # Decide whether to use original instructions or enhanced ones with checklist
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
    
    print("\n--- Starting Agent Flow Simulation ---")
    """
    Executes the agent flow with option to use structured checklist
    
    Args:
        file_path: Path to the file to be processed
        user_instructions: Specific user instructions
        focus_categories: Specific checklist categories to focus on (optional)
        min_priority: Minimum priority of checklist items ("high", "medium", "low")
        use_checklist: Whether to use the structured checklist system
    """
    print(f"Processing file: {file_path}")
    
    # Decide whether to use original instructions or enhanced ones with checklist
    if use_checklist:
        enhanced_instructions = get_enhanced_instructions(
            user_instructions=user_instructions,
            focus_categories=focus_categories,
            min_priority=min_priority
        )
        instructions_to_use = enhanced_instructions
        print("Using enhanced instructions with quality checklist")
    else:
        instructions_to_use = user_instructions
        print(f"User's overall goal: {user_instructions if user_instructions else 'No specific instructions, infer cleaning needs.'}")
    
    # Initialize the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("data_ingestion", data_ingestion_node)
    workflow.add_node("transformation_planner", transformation_planner_node)
    workflow.add_node("approval", approval_agent_node)
    workflow.add_node("code_generation", code_generation_node)  # NEW NODE
    
    # Add edges
    workflow.add_edge("data_ingestion", "transformation_planner")
    workflow.add_edge("transformation_planner", "approval")
    
    # Add conditional edge based on approval
    workflow.add_conditional_edges(
        "approval",
        route_on_approval,
        {
            "approved": "code_generation",  # If approved, go to code generation
            "rejected": "transformation_planner"
        }
    )
    
    # Set entry point
    workflow.set_entry_point("data_ingestion")
    
    # Compile the graph
    app = workflow.compile()
    
    # Run the workflow
    result = app.invoke({
        "file_path": file_path,
        "user_instructions": instructions_to_use
    })
    
    return result

# --- Example Usage Functions ---

def run_with_basic_checklist():
    """Runs with basic checklist (improved default behavior)"""
    return run_agent_flow(
        file_path="data/input_samples/precos-glp-04.csv",
        user_instructions="Clean and standardize the data for analysis",
        use_checklist=True,
        interactive_preferences=False
    )

def run_with_focused_checklist():
    """Runs with checklist focused on specific aspects"""
    return run_agent_flow(
        file_path="data/input_samples/precos-glp-04.csv",
        user_instructions="Clean and standardize the data for analysis",
        focus_categories=[DataQualityCategory.DATA_TYPES, DataQualityCategory.MISSING_VALUES],
        min_priority="high",
        use_checklist=True,
        interactive_preferences=True
    )

def run_without_checklist():
    """Default behavior: uses basic checklist"""
    return run_agent_flow(
        file_path="data/input_samples/precos-glp-04.csv",
        user_instructions="Clean and standardize the data for analysis",
        use_checklist=True,
        interactive_preferences=False
    )

# Uncomment to test other configurations:

if __name__ == "__main__":
    # Example usage
    result = run_with_basic_checklist()
    print("Workflow completed successfully!")