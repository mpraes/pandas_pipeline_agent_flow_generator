# src/utils/data_quality_checklist.py
"""
Functional checklist system for data quality.
Provides structured guiding questions for pipeline agents.
Integrated with personalized user preferences.
"""

from typing import List, NamedTuple, Optional
from enum import Enum
from .interactive_data_patterns import (
    DataCleaningPreferences, 
    get_user_preferences,
    MissingValueStrategy  # Required import
)

# === IMMUTABLE TYPES ===

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

# === CHECKLIST CREATORS BY CATEGORY ===

def create_structure_checklist() -> List[ChecklistItem]:
    """Creates checklist for data structure"""
    return [
        ChecklistItem(
            question="What is the general structure of the dataset (rows, columns, format)?",
            category=DataQualityCategory.STRUCTURE,
            priority="high",
            guidance="Understand dimensions, if there are headers, if the structure makes sense for the objective",
            examples=["Tabular dataset with 1000 rows x 15 columns", "Data in wide vs long format"]
        ),
        ChecklistItem(
            question="Are there unnecessary columns that can be removed?",
            category=DataQualityCategory.STRUCTURE,
            priority="medium",
            guidance="Identify empty, duplicate, or irrelevant columns for the objective",
            examples=["Internal ID columns", "Completely empty columns", "Audit columns"]
        ),
    ]

def create_naming_checklist() -> List[ChecklistItem]:
    """Creates checklist for naming and patterns"""
    return [
        ChecklistItem(
            question="What is the column naming pattern and how to standardize it?",
            category=DataQualityCategory.NAMING,
            priority="high",
            guidance="Define convention: snake_case, camelCase, spaces, special characters",
            examples=["user_name vs UserName vs 'User Name'", "data_nascimento vs dt_nasc"]
        ),
        ChecklistItem(
            question="Are there special characters, accents, or spaces in column names?",
            category=DataQualityCategory.NAMING,
            priority="medium",
            guidance="Consider removing accents and special characters for compatibility",
            examples=["'Preço (R$)' -> 'preco_reais'", "'Data/Hora' -> 'data_hora'"]
        ),
    ]

def create_data_types_checklist() -> List[ChecklistItem]:
    """Creates checklist for data types"""
    return [
        ChecklistItem(
            question="Are the column data types correct and optimized?",
            category=DataQualityCategory.DATA_TYPES,
            priority="high",
            guidance="Check if dates are datetime, numbers are numeric, categories are category",
            examples=["'2023-01-01' as string -> datetime", "IDs as int64 -> string"]
        ),
        ChecklistItem(
            question="Are there dates in inconsistent formats that need to be standardized?",
            category=DataQualityCategory.DATA_TYPES,
            priority="high",
            guidance="Identify different date formats and standardize to ISO format",
            examples=["01/02/2023 vs 2023-02-01 vs Feb 1, 2023"]
        ),
    ]

def create_missing_values_checklist() -> List[ChecklistItem]:
    """Creates checklist for missing values"""
    return [
        ChecklistItem(
            question="How to handle missing values in numeric columns?",
            category=DataQualityCategory.MISSING_VALUES,
            priority="high",
            guidance="Strategies: mean, median, zero, forward fill, or removal as per context",
            examples=["Null prices -> median", "Null ages -> mean", "Null IDs -> remove row"]
        ),
        ChecklistItem(
            question="How to handle missing values in categorical/text columns?",
            category=DataQualityCategory.MISSING_VALUES,
            priority="high",
            guidance="Consider: default value, 'Not informed', mode, or removal",
            examples=["Null state -> 'Not informed'", "Null category -> mode"]
        ),
        ChecklistItem(
            question="Are there implicit null representations (like 'N/A', '--', 'null')?",
            category=DataQualityCategory.MISSING_VALUES,
            priority="medium",
            guidance="Identify and convert textual null representations to NaN",
            examples=["'N/A', 'null', '--', '999999' as missing code"]
        ),
    ]

def create_text_formatting_checklist() -> List[ChecklistItem]:
    """Creates checklist for text formatting"""
    return [
        ChecklistItem(
            question="How to handle inconsistent upper/lower case strings?",
            category=DataQualityCategory.TEXT_FORMATTING,
            priority="medium",
            guidance="Standardize case according to context: proper names (Title), codes (UPPER), etc.",
            examples=["'JOÃO SILVA' vs 'joão silva' -> 'João Silva'", "States: 'sp' -> 'SP'"]
        ),
        ChecklistItem(
            question="Are there unnecessary leading/trailing spaces in texts?",
            category=DataQualityCategory.TEXT_FORMATTING,
            priority="medium",
            guidance="Remove extra spaces that can affect joins and comparisons",
            examples=["' São Paulo ' -> 'São Paulo'", "'  ABC  ' -> 'ABC'"]
        ),
        ChecklistItem(
            question="Are there incorrect control or encoding characters in texts?",
            category=DataQualityCategory.TEXT_FORMATTING,
            priority="low",
            guidance="Identify encoding issues like malformed special characters",
            examples=["'São Paulo' appearing as 'SÃ£o Paulo'"]
        ),
    ]

def create_numeric_values_checklist() -> List[ChecklistItem]:
    """Creates checklist for numeric values"""
    return [
        ChecklistItem(
            question="Are there outliers in numeric data that need to be treated?",
            category=DataQualityCategory.NUMERIC_VALUES,
            priority="medium",
            guidance="Identify extreme values: typing errors, measurements in different units",
            examples=["Age: 999 years", "Price: R$ 0,01 vs R$ 1.000.000", "Negative salary"]
        ),
        ChecklistItem(
            question="Are the numeric values in the correct scale/unit?",
            category=DataQualityCategory.NUMERIC_VALUES,
            priority="high",
            guidance="Check if values need conversion: real/cents, meters/km, etc.",
            examples=["Prices in cents vs reals", "Distances in m vs km"]
        ),
        ChecklistItem(
            question="Are there inconsistencies in decimal or thousand separators?",
            category=DataQualityCategory.NUMERIC_VALUES,
            priority="medium",
            guidance="Standardize numeric format: comma vs decimal point, thousand separators",
            examples=["1.234,56 vs 1,234.56", "1 234,56 vs 1234.56"]
        ),
    ]

def create_duplicates_checklist() -> List[ChecklistItem]:
    """Creates checklist for duplicates"""
    return [
        ChecklistItem(
            question="Are there completely duplicated records?",
            category=DataQualityCategory.DUPLICATES,
            priority="high",
            guidance="Identify and remove identical rows, keeping only one occurrence",
            examples=["Same customer registered multiple times with identical data"]
        ),
        ChecklistItem(
            question="Are there partial duplicates that need to be consolidated?",
            category=DataQualityCategory.DUPLICATES,
            priority="medium",
            guidance="Identify similar records that might be the same item with small differences",
            examples=["'João Silva' vs 'Joao Silva'", "Same product with different codes"]
        ),
    ]

def create_business_rules_checklist() -> List[ChecklistItem]:
    """Creates checklist for business rules"""
    return [
        ChecklistItem(
            question="Do the data comply with basic business rules?",
            category=DataQualityCategory.BUSINESS_RULES,
            priority="high",
            guidance="Validate logical consistency: birth date vs age, negative prices, etc.",
            examples=["Future birth date", "Negative age", "Negative price for product"]
        ),
        ChecklistItem(
            question="Are there relationships between columns that need to be validated?",
            category=DataQualityCategory.BUSINESS_RULES,
            priority="medium",
            guidance="Verify consistency between related fields",
            examples=["CEP vs City/State", "Category vs Subcategory", "Start date < End date"]
        ),
    ]

# === MAIN FUNCTIONS ===

def create_complete_checklist() -> List[ChecklistItem]:
    """
    Pure function that creates the complete checklist by combining all categories
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
    Filters checklist by priority
    Order: high > medium > low
    """
    priority_order = {"high": 3, "medium": 2, "low": 1}
    min_level = priority_order.get(min_priority, 2)
    
    return [item for item in checklist 
            if priority_order.get(item.priority, 1) >= min_level]

def filter_checklist_by_categories(checklist: List[ChecklistItem], 
                                  categories: List[DataQualityCategory]) -> List[ChecklistItem]:
    """
    Filters checklist by specific categories
    """
    if not categories:
        return checklist
    
    return [item for item in checklist if item.category in categories]

def format_checklist_as_instructions(checklist: List[ChecklistItem]) -> str:
    """
    Converts checklist to formatted string for agent instructions
    """
    instructions = ["=== DATA QUALITY CHECKLIST ===\n"]
    
    # Groups by category
    categories = {}
    for item in checklist:
        if item.category not in categories:
            categories[item.category] = []
        categories[item.category].append(item)
    
    for category, items in categories.items():
        instructions.append(f"## {category.value.upper().replace('_', ' ')}")
        for i, item in enumerate(items, 1):
            instructions.append(f"{i}. **{item.question}**")
            instructions.append(f"   - Guidance: {item.guidance}")
            instructions.append(f"   - Priority: {item.priority}")
            if item.examples:
                instructions.append(f"   - Examples: {'; '.join(item.examples)}")
            instructions.append("")
        instructions.append("")
    
    return "\n".join(instructions)

def create_quality_instructions(user_instructions: Optional[str] = None,
                              focus_categories: List[DataQualityCategory] = None,
                              min_priority: str = "medium") -> QualityInstructions:
    """
    Main function that creates complete instructions for agents
    """
    complete_checklist = create_complete_checklist()
    
    # Applies filters
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
    Returns complete formatted instructions for agents,
    including personalized user preferences.
    
    Args:
        user_instructions: Basic user instructions
        focus_categories: Specific checklist categories
        min_priority: Minimum priority for items
        use_interactive_preferences: If it should collect preferences interactively
    """
    # Variable to store preferences in the current session
    session_preferences = getattr(get_enhanced_instructions, '_session_preferences', None)
    
    # Collect user preferences if requested and not already collected
    user_preferences = None
    if use_interactive_preferences:
        if session_preferences is None:
            user_preferences = get_user_preferences(use_saved=True)
            # Store preferences in the session to avoid re-asking
            get_enhanced_instructions._session_preferences = user_preferences
        else:
            user_preferences = session_preferences
    
    # Creates default checklist
    quality_instructions = create_quality_instructions(
        user_instructions, focus_categories, min_priority
    )
    
    formatted_checklist = format_checklist_as_instructions(quality_instructions.checklist)
    
    final_instructions = []
    
    if user_instructions:
        final_instructions.append("=== USER INSTRUCTIONS ===")
        final_instructions.append(user_instructions)
        final_instructions.append("")
    
    # Adds personalized preferences if collected
    if user_preferences:
        final_instructions.append("=== PERSONALIZED CLEANING PATTERNS ===")
        final_instructions.append(_format_user_preferences(user_preferences))
        final_instructions.append("")
    
    final_instructions.append(formatted_checklist)
    final_instructions.append("=== GENERAL ORIENTATIONS ===")
    
    if user_preferences:
        final_instructions.append("- Apply the user-defined personalized patterns")
        final_instructions.append("- Use the preferences as a guide for all cleaning decisions")
    else:
        final_instructions.append("- Use sensible cleaning standards and widely accepted ones")
        final_instructions.append("- Prioritize consistency and compatibility")
    
    final_instructions.append("- Systematically analyze each checklist question")
    final_instructions.append("- Prioritize high-priority items")
    final_instructions.append("- Justify your decisions based on the context of the data")
    final_instructions.append("- Consider the final objective of the analysis")
    
    return "\n".join(final_instructions)

def _format_user_preferences(preferences: DataCleaningPreferences) -> str:
    """
    Formats user preferences into readable text for agents.
    
    Args:
        preferences: User cleaning preferences
        
    Returns:
        Formatted string with preferences
    """
    prefs_text = []
    
    # Column naming preferences
    prefs_text.append("**COLUMN NAMING:**")
    prefs_text.append(f"- Style: {preferences.column_naming_style.value}")
    if preferences.remove_column_spaces:
        prefs_text.append("- Remove spaces from column names")
    if preferences.remove_column_accents:
        prefs_text.append("- Remove accents from column names")
    if preferences.remove_column_special_chars:
        prefs_text.append("- Remove special characters from column names")
    
    # String formatting preferences  
    prefs_text.append("\n**STRING FORMATTING:**")
    prefs_text.append(f"- Case style: {preferences.string_case_style.value}")
    prefs_text.append(f"- Cleaning level: {preferences.string_cleaning_level.value}")
    if preferences.remove_leading_trailing_spaces:
        prefs_text.append("- Remove leading/trailing spaces")
    if preferences.normalize_whitespace:
        prefs_text.append("- Normalize whitespace")
    
    # Numeric formatting preferences
    prefs_text.append("\n**NUMERIC FORMATTING:**")
    prefs_text.append(f"- Decimal format: {preferences.numeric_format.value}")
    if preferences.decimal_places is not None:
        prefs_text.append(f"- Default decimal places: {preferences.decimal_places}")
    if preferences.remove_currency_symbols:
        prefs_text.append("- Remove currency symbols")
    if preferences.handle_thousand_separators:
        prefs_text.append("- Handle thousand separators")
    
    # Date formatting preferences
    prefs_text.append("\n**DATE FORMATTING:**")
    prefs_text.append(f"- Preferred format: {preferences.date_format.value}")
    if preferences.standardize_date_format:
        prefs_text.append("- Standardize all dates")
    prefs_text.append(f"- Invalid dates: {preferences.handle_invalid_dates}")
    
    # Missing value preferences
    prefs_text.append("\n**MISSING VALUES:**")
    prefs_text.append(f"- Numeric strategy: {preferences.numeric_missing_strategy.value}")
    prefs_text.append(f"- Categorical strategy: {preferences.categorical_missing_strategy.value}")
    if preferences.categorical_missing_strategy in [MissingValueStrategy.FILL_CUSTOM]:
        prefs_text.append(f"- Value for missing categorical: '{preferences.categorical_fill_value}'")
    prefs_text.append(f"- Remove columns with >{preferences.missing_threshold*100:.0f}% missing values")
    
    # General preferences
    prefs_text.append("\n**GENERAL PREFERENCES:**")
    if preferences.remove_duplicates:
        prefs_text.append("- Remove duplicates automatically")
    if preferences.handle_outliers:
        prefs_text.append("- Detect and treat outliers")
    if preferences.encoding_fix:
        prefs_text.append("- Fix encoding issues")
    if preferences.memory_optimization:
        prefs_text.append("- Optimize data types for memory")
    
    return "\n".join(prefs_text)

def clear_session_preferences():
    """
    Clears current session preferences.
    Useful to force new preference collection.
    """
    if hasattr(get_enhanced_instructions, '_session_preferences'):
        delattr(get_enhanced_instructions, '_session_preferences')
        print("Session preferences cleared.")