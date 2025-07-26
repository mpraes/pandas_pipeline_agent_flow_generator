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
from src.agents.code_generation_agent import code_generation_node  # NOVO IMPORT
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
    Executa o fluxo de agentes com opção de usar checklist estruturado e preferências interativas
    
    Args:
        file_path: Caminho para o arquivo a ser processado
        user_instructions: Instruções específicas do usuário
        focus_categories: Categorias específicas do checklist para focar (opcional)
        min_priority: Prioridade mínima dos itens do checklist ("high", "medium", "low")
        use_checklist: Se deve usar o sistema de checklist estruturado
        interactive_preferences: Se deve coletar preferências interativamente
    """
    print(f"Processing file: {file_path}")
    
    # Decide se usa as instruções originais ou aprimoradas com checklist
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
    Executa o fluxo de agentes com opção de usar checklist estruturado
    
    Args:
        file_path: Caminho para o arquivo a ser processado
        user_instructions: Instruções específicas do usuário
        focus_categories: Categorias específicas do checklist para focar (opcional)
        min_priority: Prioridade mínima dos itens do checklist ("high", "medium", "low")
        use_checklist: Se deve usar o sistema de checklist estruturado
    """
    print(f"Processing file: {file_path}")
    
    # Decide se usa as instruções originais ou aprimoradas com checklist
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
    
    print("\n--- Starting Agent Flow Simulation ---")

    workflow = StateGraph(GraphState)

    workflow.add_node("data_ingestion", data_ingestion_node)
    workflow.add_node("transformation_planner", transformation_planner_node)
    workflow.add_node("approval_agent", approval_agent_node)
    workflow.add_node("code_generation", code_generation_node)  # NOVO NÓ

    # Define the flow (edges)
    workflow.set_entry_point("data_ingestion")
    workflow.add_edge("data_ingestion", "transformation_planner")
    workflow.add_edge("transformation_planner", "approval_agent")

    # Add conditional edge based on user approval
    workflow.add_conditional_edges(
        "approval_agent",
        route_on_approval,
        {
            "approved": "code_generation",  # Se aprovado, vai para geração de código
            "rejected": "transformation_planner" # Se rejeitado, volta para o planejador
        }
    )
    
    # Code generation goes to END
    workflow.add_edge("code_generation", END)

    app = workflow.compile()

    final_state = app.invoke({
        "file_path": file_path,
        "source_type": "csv",
        "user_instructions": instructions_to_use
    })

    print("\n--- Agent Flow Simulation Finished ---")
    
    if final_state.get("transformation_plan"):
        print("\n--- Final Transformation Plan Status ---")
        print(f"Plan Approved: {final_state['user_approval'].approved}")
        if final_state['user_approval'].feedback:
            print(f"Feedback: {final_state['user_approval'].feedback}")
        
        # Check if code was generated
        if final_state.get("generated_code"):
            print("\n--- Code Generation Results ---")
            print(f"✅ Generated executable pandas code")
            if final_state.get("code_file_path"):
                print(f"📁 Code saved to: {final_state['code_file_path']}")
            print(f"📝 Code length: {len(final_state['generated_code'])} characters")
            print(f"📏 Lines of code: {len(final_state['generated_code'].split('\\n'))}")
            
            # Show code preview
            code_lines = final_state['generated_code'].split('\\n')
            print("\\n--- Code Preview (first 20 lines) ---")
            for i, line in enumerate(code_lines[:20], 1):
                print(f"{i:2d}: {line}")
            if len(code_lines) > 20:
                print(f"... and {len(code_lines) - 20} more lines")
        
        print("\\n--- Generated Transformation Plan (full details) ---")
        print(final_state["transformation_plan"].model_dump_json(indent=2))
    elif final_state.get("error_message"):
        print(f"\\n--- Planner Error: {final_state['error_message']} ---")

# === EXEMPLOS DE CONFIGURAÇÃO ===

def run_with_basic_checklist():
    """Executa com checklist básico (comportamento padrão melhorado)"""
    input_file = "data/input_samples/precos-glp-04.csv"
    general_instructions = "Clean and standardize the data for analysis and reporting, focusing on common data quality issues and usability."
    
    run_agent_flow(input_file, general_instructions)

def run_with_focused_checklist():
    """Executa com checklist focado em aspectos específicos"""
    input_file = "data/input_samples/precos-glp-04.csv"
    specific_instructions = "Prepare data for financial analysis with focus on numeric consistency."
    
    # Foca apenas em categorias relevantes para análise financeira
    focus_categories = [
        DataQualityCategory.NUMERIC_VALUES,
        DataQualityCategory.DATA_TYPES,
        DataQualityCategory.MISSING_VALUES
    ]
    
    run_agent_flow(
        file_path=input_file,
        user_instructions=specific_instructions,
        focus_categories=focus_categories,
        min_priority="high"
    )

def run_without_checklist():
    """Executa sem checklist (comportamento original)"""
    input_file = "data/input_samples/precos-glp-04.csv"
    general_instructions = "Clean and standardize the data for analysis and reporting, focusing on common data quality issues and usability."
    
    run_agent_flow(input_file, general_instructions, use_checklist=False)

if __name__ == "__main__":
    # Comportamento padrão: usa checklist básico
    input_file = "data/input_samples/precos-glp-04.csv"
    general_instructions = "Clean and standardize the data for analysis and reporting, focusing on common data quality issues and usability."
    
    run_agent_flow(input_file, general_instructions)
    
    # Descomente para testar outras configurações:
    
    # print("\n\n=== Executando com Checklist Focado ===")
    # run_with_focused_checklist()
    
    # print("\n\n=== Executando sem Checklist (Original) ===")
    # run_without_checklist()