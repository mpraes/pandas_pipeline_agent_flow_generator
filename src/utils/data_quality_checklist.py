# src/utils/data_quality_checklist.py
"""
Sistema de checklist funcional para qualidade de dados.
Fornece perguntas orientadoras estruturadas para os agentes de pipeline.
Integrado com preferências personalizadas do usuário.
"""

from typing import List, NamedTuple, Optional
from enum import Enum
from .interactive_data_patterns import (
    DataCleaningPreferences, 
    get_user_preferences,
    MissingValueStrategy  # Import necessário
)

# === TIPOS IMUTÁVEIS ===

class DataQualityCategory(Enum):
    STRUCTURE = "structure"
    NAMING = "naming"
    DATA_TYPES = "data_types" 
    MISSING_VALUES = "missing_values"
    TEXT_FORMATTING = "text_formatting"
    NUMERIC_VALUES = "numeric_values"
    DUPLICATES = "duplicates"
    BUSINESS_RULES = "business_rules"

class ChecklistItem(NamedTuple):
    question: str
    category: DataQualityCategory
    priority: str  # "high", "medium", "low"
    guidance: str
    examples: List[str]

class QualityInstructions(NamedTuple):
    checklist: List[ChecklistItem]
    user_instructions: Optional[str]
    focus_categories: List[DataQualityCategory]

# === CRIADORES DE CHECKLIST POR CATEGORIA ===

def create_structure_checklist() -> List[ChecklistItem]:
    """Cria checklist para estrutura dos dados"""
    return [
        ChecklistItem(
            question="Qual é a estrutura geral do dataset (linhas, colunas, formato)?",
            category=DataQualityCategory.STRUCTURE,
            priority="high",
            guidance="Entenda dimensões, se há cabeçalhos, se a estrutura faz sentido para o objetivo",
            examples=["Dataset tabular com 1000 linhas x 15 colunas", "Dados em formato wide vs long"]
        ),
        ChecklistItem(
            question="Existem colunas desnecessárias ou que podem ser removidas?",
            category=DataQualityCategory.STRUCTURE,
            priority="medium",
            guidance="Identifique colunas vazias, duplicadas ou irrelevantes para o objetivo",
            examples=["Colunas de ID interno", "Colunas completamente vazias", "Colunas de auditoria"]
        ),
    ]

def create_naming_checklist() -> List[ChecklistItem]:
    """Cria checklist para nomenclatura e padrões"""
    return [
        ChecklistItem(
            question="Qual o padrão de nomenclatura das colunas e como padronizar?",
            category=DataQualityCategory.NAMING,
            priority="high",
            guidance="Defina convenção: snake_case, camelCase, espaços, caracteres especiais",
            examples=["user_name vs UserName vs 'User Name'", "data_nascimento vs dt_nasc"]
        ),
        ChecklistItem(
            question="Há caracteres especiais, acentos ou espaços nos nomes das colunas?",
            category=DataQualityCategory.NAMING,
            priority="medium",
            guidance="Considere remover acentos e caracteres especiais para compatibilidade",
            examples=["'Preço (R$)' -> 'preco_reais'", "'Data/Hora' -> 'data_hora'"]
        ),
    ]

def create_data_types_checklist() -> List[ChecklistItem]:
    """Cria checklist para tipos de dados"""
    return [
        ChecklistItem(
            question="Os tipos de dados das colunas estão corretos e otimizados?",
            category=DataQualityCategory.DATA_TYPES,
            priority="high",
            guidance="Verifique se datas são datetime, números são numeric, categorias são category",
            examples=["'2023-01-01' como string -> datetime", "IDs como int64 -> string"]
        ),
        ChecklistItem(
            question="Existem datas em formatos inconsistentes que precisam ser padronizadas?",
            category=DataQualityCategory.DATA_TYPES,
            priority="high",
            guidance="Identifique diferentes formatos de data e padronize para ISO format",
            examples=["01/02/2023 vs 2023-02-01 vs Feb 1, 2023"]
        ),
    ]

def create_missing_values_checklist() -> List[ChecklistItem]:
    """Cria checklist para valores ausentes"""
    return [
        ChecklistItem(
            question="Como lidar com valores nulos/ausentes em colunas numéricas?",
            category=DataQualityCategory.MISSING_VALUES,
            priority="high",
            guidance="Estratégias: média, mediana, zero, forward fill, ou remoção conforme contexto",
            examples=["Preços nulos -> mediana", "Idades nulas -> média", "IDs nulos -> remover linha"]
        ),
        ChecklistItem(
            question="Como tratar valores ausentes em colunas categóricas/texto?",
            category=DataQualityCategory.MISSING_VALUES,
            priority="high",
            guidance="Considere: valor padrão, 'Não informado', moda, ou remoção",
            examples=["Estado nulo -> 'Não informado'", "Categoria nula -> moda"]
        ),
        ChecklistItem(
            question="Há representações implícitas de nulos (como 'N/A', '--', 'null')?",
            category=DataQualityCategory.MISSING_VALUES,
            priority="medium",
            guidance="Identifique e converta representações textuais de nulos para NaN",
            examples=["'N/A', 'null', '--', '999999' como código de ausente"]
        ),
    ]

def create_text_formatting_checklist() -> List[ChecklistItem]:
    """Cria checklist para formatação de texto"""
    return [
        ChecklistItem(
            question="Como lidar com strings em upper/lower case inconsistentes?",
            category=DataQualityCategory.TEXT_FORMATTING,
            priority="medium",
            guidance="Padronize case conforme contexto: nomes próprios (Title), códigos (UPPER), etc.",
            examples=["'JOÃO SILVA' vs 'joão silva' -> 'João Silva'", "Estados: 'sp' -> 'SP'"]
        ),
        ChecklistItem(
            question="Existem espaços em branco desnecessários no início/fim dos textos?",
            category=DataQualityCategory.TEXT_FORMATTING,
            priority="medium",
            guidance="Remova espaços extras que podem afetar joins e comparações",
            examples=["' São Paulo ' -> 'São Paulo'", "'  ABC  ' -> 'ABC'"]
        ),
        ChecklistItem(
            question="Há caracteres de controle ou encoding incorreto nos textos?",
            category=DataQualityCategory.TEXT_FORMATTING,
            priority="low",
            guidance="Identifique problemas de encoding como caracteres especiais malformados",
            examples=["'São Paulo' aparecendo como 'SÃ£o Paulo'"]
        ),
    ]

def create_numeric_values_checklist() -> List[ChecklistItem]:
    """Cria checklist para valores numéricos"""
    return [
        ChecklistItem(
            question="Existem outliers nos dados numéricos que precisam ser tratados?",
            category=DataQualityCategory.NUMERIC_VALUES,
            priority="medium",
            guidance="Identifique valores extremos: erros de digitação, medidas em unidades diferentes",
            examples=["Idade: 999 anos", "Preço: R$ 0,01 vs R$ 1.000.000", "Salário negativo"]
        ),
        ChecklistItem(
            question="Os valores numéricos estão na escala/unidade correta?",
            category=DataQualityCategory.NUMERIC_VALUES,
            priority="high",
            guidance="Verifique se valores precisam conversão: reais/centavos, metros/km, etc.",
            examples=["Preços em centavos vs reais", "Distâncias em m vs km"]
        ),
        ChecklistItem(
            question="Há inconsistências em separadores decimais ou milhares?",
            category=DataQualityCategory.NUMERIC_VALUES,
            priority="medium",
            guidance="Padronize formato numérico: vírgula vs ponto decimal, separadores de milhares",
            examples=["1.234,56 vs 1,234.56", "1 234,56 vs 1234.56"]
        ),
    ]

def create_duplicates_checklist() -> List[ChecklistItem]:
    """Cria checklist para duplicatas"""
    return [
        ChecklistItem(
            question="Existem registros completamente duplicados?",
            category=DataQualityCategory.DUPLICATES,
            priority="high",
            guidance="Identifique e remova linhas idênticas, mantendo apenas uma ocorrência",
            examples=["Mesmo cliente cadastrado múltiplas vezes com dados idênticos"]
        ),
        ChecklistItem(
            question="Há duplicatas parciais que precisam ser consolidadas?",
            category=DataQualityCategory.DUPLICATES,
            priority="medium",
            guidance="Identifique registros similares que podem ser o mesmo item com pequenas diferenças",
            examples=["'João Silva' vs 'Joao Silva'", "Mesmo produto com códigos diferentes"]
        ),
    ]

def create_business_rules_checklist() -> List[ChecklistItem]:
    """Cria checklist para regras de negócio"""
    return [
        ChecklistItem(
            question="Os dados respeitam as regras de negócio básicas?",
            category=DataQualityCategory.BUSINESS_RULES,
            priority="high",
            guidance="Valide consistência lógica: datas de nascimento vs idade, preços negativos, etc.",
            examples=["Data nascimento futura", "Idade negativa", "Preço negativo para produto"]
        ),
        ChecklistItem(
            question="Existem relacionamentos entre colunas que devem ser validados?",
            category=DataQualityCategory.BUSINESS_RULES,
            priority="medium",
            guidance="Verifique consistência entre campos relacionados",
            examples=["CEP vs Cidade/Estado", "Categoria vs Subcategoria", "Data início < Data fim"]
        ),
    ]

# === FUNÇÕES PRINCIPAIS ===

def create_complete_checklist() -> List[ChecklistItem]:
    """
    Função pura que cria o checklist completo combinando todas as categorias
    """
    checklist_creators = [
        create_structure_checklist,
        create_naming_checklist,
        create_data_types_checklist,
        create_missing_values_checklist,
        create_text_formatting_checklist,
        create_numeric_values_checklist,
        create_duplicates_checklist,
        create_business_rules_checklist,
    ]
    
    return [item for creator in checklist_creators for item in creator()]

def filter_checklist_by_priority(checklist: List[ChecklistItem], 
                                min_priority: str = "medium") -> List[ChecklistItem]:
    """
    Filtra checklist por prioridade
    Ordem: high > medium > low
    """
    priority_order = {"high": 3, "medium": 2, "low": 1}
    min_level = priority_order.get(min_priority, 2)
    
    return [item for item in checklist 
            if priority_order.get(item.priority, 1) >= min_level]

def filter_checklist_by_categories(checklist: List[ChecklistItem], 
                                  categories: List[DataQualityCategory]) -> List[ChecklistItem]:
    """
    Filtra checklist por categorias específicas
    """
    if not categories:
        return checklist
    
    return [item for item in checklist if item.category in categories]

def format_checklist_as_instructions(checklist: List[ChecklistItem]) -> str:
    """
    Converte checklist em string formatada para instruções dos agentes
    """
    instructions = ["=== CHECKLIST DE QUALIDADE DE DADOS ===\n"]
    
    # Agrupa por categoria
    categories = {}
    for item in checklist:
        if item.category not in categories:
            categories[item.category] = []
        categories[item.category].append(item)
    
    for category, items in categories.items():
        instructions.append(f"## {category.value.upper().replace('_', ' ')}")
        for i, item in enumerate(items, 1):
            instructions.append(f"{i}. **{item.question}**")
            instructions.append(f"   - Orientação: {item.guidance}")
            instructions.append(f"   - Prioridade: {item.priority}")
            if item.examples:
                instructions.append(f"   - Exemplos: {'; '.join(item.examples)}")
            instructions.append("")
        instructions.append("")
    
    return "\n".join(instructions)

def create_quality_instructions(user_instructions: Optional[str] = None,
                              focus_categories: List[DataQualityCategory] = None,
                              min_priority: str = "medium") -> QualityInstructions:
    """
    Função principal que cria as instruções completas para os agentes
    """
    complete_checklist = create_complete_checklist()
    
    # Aplica filtros
    filtered_checklist = filter_checklist_by_priority(complete_checklist, min_priority)
    if focus_categories:
        filtered_checklist = filter_checklist_by_categories(filtered_checklist, focus_categories)
    
    return QualityInstructions(
        checklist=filtered_checklist,
        user_instructions=user_instructions,
        focus_categories=focus_categories or []
    )

def get_enhanced_instructions(user_instructions: Optional[str] = None,
                            focus_categories: List[DataQualityCategory] = None,
                            min_priority: str = "medium",
                            use_interactive_preferences: bool = True) -> str:
    """
    Retorna as instruções completas formatadas para os agentes,
    incluindo preferências personalizadas do usuário.
    
    Args:
        user_instructions: Instruções básicas do usuário
        focus_categories: Categorias específicas do checklist
        min_priority: Prioridade mínima dos itens
        use_interactive_preferences: Se deve coletar preferências interativamente
    """
    # Variável para armazenar preferências na sessão atual
    session_preferences = getattr(get_enhanced_instructions, '_session_preferences', None)
    
    # Coleta preferências do usuário se solicitado e ainda não foram coletadas
    user_preferences = None
    if use_interactive_preferences:
        if session_preferences is None:
            user_preferences = get_user_preferences(use_saved=True)
            # Armazena as preferências na sessão para evitar re-perguntar
            get_enhanced_instructions._session_preferences = user_preferences
        else:
            user_preferences = session_preferences
    
    # Cria checklist padrão
    quality_instructions = create_quality_instructions(
        user_instructions, focus_categories, min_priority
    )
    
    formatted_checklist = format_checklist_as_instructions(quality_instructions.checklist)
    
    final_instructions = []
    
    if user_instructions:
        final_instructions.append("=== INSTRUÇÕES DO USUÁRIO ===")
        final_instructions.append(user_instructions)
        final_instructions.append("")
    
    # Adiciona preferências personalizadas se coletadas
    if user_preferences:
        final_instructions.append("=== PADRÕES DE LIMPEZA PERSONALIZADOS ===")
        final_instructions.append(_format_user_preferences(user_preferences))
        final_instructions.append("")
    
    final_instructions.append(formatted_checklist)
    final_instructions.append("=== ORIENTAÇÕES GERAIS ===")
    
    if user_preferences:
        final_instructions.append("- Aplique os padrões personalizados definidos pelo usuário")
        final_instructions.append("- Use as preferências como guia para todas as decisões de limpeza")
    else:
        final_instructions.append("- Use padrões de limpeza sensatos e amplamente aceitos")
        final_instructions.append("- Priorize consistência e compatibilidade")
    
    final_instructions.append("- Analise cada pergunta do checklist sistematicamente")
    final_instructions.append("- Priorize itens de alta prioridade")
    final_instructions.append("- Justifique suas decisões baseado no contexto dos dados")
    final_instructions.append("- Considere o objetivo final da análise")
    
    return "\n".join(final_instructions)

def _format_user_preferences(preferences: DataCleaningPreferences) -> str:
    """
    Formata as preferências do usuário em texto legível para os agentes.
    
    Args:
        preferences: Preferências de limpeza do usuário
        
    Returns:
        String formatada com as preferências
    """
    prefs_text = []
    
    # Column naming preferences
    prefs_text.append("**NOMENCLATURA DE COLUNAS:**")
    prefs_text.append(f"- Estilo: {preferences.column_naming_style.value}")
    if preferences.remove_column_spaces:
        prefs_text.append("- Remover espaços dos nomes das colunas")
    if preferences.remove_column_accents:
        prefs_text.append("- Remover acentos dos nomes das colunas")
    if preferences.remove_column_special_chars:
        prefs_text.append("- Remover caracteres especiais dos nomes das colunas")
    
    # String formatting preferences  
    prefs_text.append("\n**FORMATAÇÃO DE STRINGS:**")
    prefs_text.append(f"- Estilo de case: {preferences.string_case_style.value}")
    prefs_text.append(f"- Nível de limpeza: {preferences.string_cleaning_level.value}")
    if preferences.remove_leading_trailing_spaces:
        prefs_text.append("- Remover espaços no início/fim")
    if preferences.normalize_whitespace:
        prefs_text.append("- Normalizar espaços em branco")
    
    # Numeric formatting preferences
    prefs_text.append("\n**FORMATAÇÃO NUMÉRICA:**")
    prefs_text.append(f"- Formato decimal: {preferences.numeric_format.value}")
    if preferences.decimal_places is not None:
        prefs_text.append(f"- Casas decimais padrão: {preferences.decimal_places}")
    if preferences.remove_currency_symbols:
        prefs_text.append("- Remover símbolos de moeda")
    if preferences.handle_thousand_separators:
        prefs_text.append("- Tratar separadores de milhares")
    
    # Date formatting preferences
    prefs_text.append("\n**FORMATAÇÃO DE DATAS:**")
    prefs_text.append(f"- Formato preferido: {preferences.date_format.value}")
    if preferences.standardize_date_format:
        prefs_text.append("- Padronizar todas as datas")
    prefs_text.append(f"- Datas inválidas: {preferences.handle_invalid_dates}")
    
    # Missing value preferences
    prefs_text.append("\n**VALORES AUSENTES:**")
    prefs_text.append(f"- Estratégia numérica: {preferences.numeric_missing_strategy.value}")
    prefs_text.append(f"- Estratégia categórica: {preferences.categorical_missing_strategy.value}")
    if preferences.categorical_missing_strategy in [MissingValueStrategy.FILL_CUSTOM]:
        prefs_text.append(f"- Valor para categóricos ausentes: '{preferences.categorical_fill_value}'")
    prefs_text.append(f"- Remover colunas com >{preferences.missing_threshold*100:.0f}% de valores ausentes")
    
    # General preferences
    prefs_text.append("\n**PREFERÊNCIAS GERAIS:**")
    if preferences.remove_duplicates:
        prefs_text.append("- Remover duplicatas automaticamente")
    if preferences.handle_outliers:
        prefs_text.append("- Detectar e tratar outliers")
    if preferences.encoding_fix:
        prefs_text.append("- Corrigir problemas de encoding")
    if preferences.memory_optimization:
        prefs_text.append("- Otimizar tipos de dados para memória")
    
    return "\n".join(prefs_text)

def clear_session_preferences():
    """
    Limpa as preferências da sessão atual.
    Útil para forçar nova coleta de preferências.
    """
    if hasattr(get_enhanced_instructions, '_session_preferences'):
        delattr(get_enhanced_instructions, '_session_preferences')
        print("Session preferences cleared.")