# src/agents/approval_agent.py

from typing import Dict, Any
from src.core.data_schema import GraphState, TransformationPlan, UserApproval # Import UserApproval
from src.core.llm_config import get_llm # You might use LLM to summarize the plan for the user

def approval_agent_node(state: GraphState) -> Dict[str, Any]:
    """
    Agent responsible for requesting user approval for the transformation plan.

    Args:
        state: The current graph state, expected to contain 'transformation_plan'.

    Returns:
        A dictionary containing 'user_approval' (UserApproval object).
    """
    transformation_plan: TransformationPlan = state["transformation_plan"]

    print("\n" + "="*50)
    print("--- USER APPROVAL REQUIRED ---")
    print("="*50)
    print("\nI've generated a data transformation plan based on your data and goals.")
    print("Please review it carefully.\n")

    # Display a summary of the plan
    print("--- Transformation Plan Summary ---")
    print(transformation_plan.overall_summary)
    print("\nDetailed Steps:")
    for i, step in enumerate(transformation_plan.transformation_steps):
        print(f"  {i+1}. Operation Type: {step.operation_type.value}")
        print(f"     Column: {step.column_name if step.column_name else 'N/A'}")
        print(f"     Justification: {step.justification}")
        print(f"     Expected Outcome: {step.expected_outcome}")
        if step.params:
            print(f"     Parameters: {step.params}")
        print("-" * 30)

    # Ask for user confirmation
    user_response_str = input("\nDo you approve this transformation plan? (yes/no): ").strip().lower()

    if user_response_str == "yes":
        print("\nUser approved the plan. Proceeding with data transformation.")
        user_approval = UserApproval(approved=True, feedback="Plan approved by user.")
    else:
        feedback = input("Please provide feedback for rejection (optional): ").strip()
        print("\nUser rejected the plan. Flow terminated or will be re-evaluated.")
        user_approval = UserApproval(approved=False, feedback=feedback if feedback else "Plan rejected without specific feedback.")
    
    # Return the UserApproval object to update the graph state
    return {"user_approval": user_approval}