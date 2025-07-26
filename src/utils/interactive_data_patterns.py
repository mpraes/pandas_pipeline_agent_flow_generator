# src/utils/interactive_data_patterns.py
"""
Interactive system to capture user preferences for data cleaning patterns.
Defines standards for column naming, string formatting, numeric handling, etc.
"""

from typing import Dict, List, NamedTuple, Optional
from enum import Enum
import json

# === DATA CLEANING PATTERN TYPES ===

class ColumnNamingStyle(Enum):
    SNAKE_CASE = "snake_case"           # exemplo_coluna
    KEBAB_CASE = "kebab-case"           # exemplo-coluna  
    CAMEL_CASE = "camelCase"            # exemploColuna
    PASCAL_CASE = "PascalCase"          # ExemploColuna
    LOWER_UNDERSCORE = "lower_underscore" # exemplo_coluna (same as snake but explicit)
    KEEP_ORIGINAL = "keep_original"     # Mantém formato original

class StringCaseStyle(Enum):
    LOWER = "lower"                     # texto em minúsculo
    UPPER = "upper"                     # TEXTO EM MAIÚSCULO  
    TITLE = "title"                     # Texto Em Título
    SENTENCE = "sentence"               # Texto em sentença
    KEEP_ORIGINAL = "keep_original"     # Mantém formato original

class NumericFormat(Enum):
    DECIMAL_DOT = "decimal_dot"         # 1234.56 (padrão americano)
    DECIMAL_COMMA = "decimal_comma"     # 1234,56 (padrão brasileiro)
    AUTO_DETECT = "auto_detect"         # Detecta automaticamente
    KEEP_ORIGINAL = "keep_original"     # Mantém formato original

class DateFormat(Enum):
    ISO_FORMAT = "iso_format"           # YYYY-MM-DD
    US_FORMAT = "us_format"             # MM/DD/YYYY
    BR_FORMAT = "br_format"             # DD/MM/YYYY
    AUTO_DETECT = "auto_detect"         # Detecta automaticamente
    KEEP_ORIGINAL = "keep_original"     # Mantém formato original

class MissingValueStrategy(Enum):
    DROP_ROWS = "drop_rows"             # Remove linhas com valores ausentes
    FILL_ZERO = "fill_zero"             # Preenche com 0
    FILL_MEAN = "fill_mean"             # Preenche com média (numérico)
    FILL_MEDIAN = "fill_median"         # Preenche com mediana (numérico)
    FILL_MODE = "fill_mode"             # Preenche com moda (categórico)
    FILL_FORWARD = "fill_forward"       # Forward fill
    FILL_BACKWARD = "fill_backward"     # Backward fill
    FILL_CUSTOM = "fill_custom"         # Valor customizado
    KEEP_AS_NULL = "keep_as_null"       # Mantém como nulo

class TextCleaningLevel(Enum):
    MINIMAL = "minimal"                 # Remove apenas espaços extras
    MODERATE = "moderate"               # Remove espaços, caracteres especiais básicos
    AGGRESSIVE = "aggressive"           # Limpeza completa, remove acentos, etc.
    CUSTOM = "custom"                   # Configuração customizada

# === DATA CLEANING PREFERENCES ===

class DataCleaningPreferences(NamedTuple):
    """Complete set of user preferences for data cleaning patterns"""
    
    # Column naming preferences
    column_naming_style: ColumnNamingStyle
    remove_column_spaces: bool
    remove_column_accents: bool
    remove_column_special_chars: bool
    
    # String formatting preferences  
    string_case_style: StringCaseStyle
    string_cleaning_level: TextCleaningLevel
    remove_leading_trailing_spaces: bool
    normalize_whitespace: bool
    
    # Numeric formatting preferences
    numeric_format: NumericFormat
    decimal_places: Optional[int]
    remove_currency_symbols: bool
    handle_thousand_separators: bool
    
    # Date formatting preferences
    date_format: DateFormat
    standardize_date_format: bool
    handle_invalid_dates: str  # 'drop', 'fill_today', 'fill_custom'
    
    # Missing value preferences
    numeric_missing_strategy: MissingValueStrategy
    categorical_missing_strategy: MissingValueStrategy
    categorical_fill_value: str  # Custom value for categorical missing data
    missing_threshold: float  # % of missing values to consider dropping column
    
    # General preferences
    remove_duplicates: bool
    handle_outliers: bool
    encoding_fix: bool
    memory_optimization: bool

# === INTERACTIVE PREFERENCE COLLECTION ===

def collect_user_preferences() -> DataCleaningPreferences:
    """
    Interactive function to collect user preferences for data cleaning patterns.
    
    Returns:
        DataCleaningPreferences object with user's choices
    """
    print("\n" + "="*60)
    print("🔧 DATA CLEANING PATTERN CONFIGURATION")
    print("="*60)
    print("Let's configure your preferred data cleaning patterns.")
    print("These settings will be applied to all datasets you process.\n")
    
    # Column naming preferences
    print("📋 COLUMN NAMING PREFERENCES")
    print("-" * 30)
    column_styles = {
        "1": ColumnNamingStyle.SNAKE_CASE,
        "2": ColumnNamingStyle.KEBAB_CASE, 
        "3": ColumnNamingStyle.CAMEL_CASE,
        "4": ColumnNamingStyle.PASCAL_CASE,
        "5": ColumnNamingStyle.KEEP_ORIGINAL
    }
    
    print("Column naming style:")
    print("1. snake_case (recommended) - exemplo_coluna")
    print("2. kebab-case - exemplo-coluna") 
    print("3. camelCase - exemploColuna")
    print("4. PascalCase - ExemploColuna")
    print("5. Keep original format")
    
    column_choice = input("Choose column naming style [1-5] (default: 1): ").strip() or "1"
    column_naming_style = column_styles.get(column_choice, ColumnNamingStyle.SNAKE_CASE)
    
    remove_spaces = input("Remove spaces from column names? [y/N]: ").strip().lower() == 'y'
    remove_accents = input("Remove accents from column names? [y/N]: ").strip().lower() == 'y'
    remove_special = input("Remove special characters from column names? [y/N]: ").strip().lower() == 'y'
    
    # String formatting preferences
    print("\n📝 STRING FORMATTING PREFERENCES")
    print("-" * 35)
    string_styles = {
        "1": StringCaseStyle.LOWER,
        "2": StringCaseStyle.UPPER,
        "3": StringCaseStyle.TITLE,
        "4": StringCaseStyle.SENTENCE,
        "5": StringCaseStyle.KEEP_ORIGINAL
    }
    
    print("Default string case style:")
    print("1. lowercase - texto em minúsculo")
    print("2. UPPERCASE - TEXTO EM MAIÚSCULO")
    print("3. Title Case - Texto Em Título")
    print("4. Sentence case - Texto em sentença")
    print("5. Keep original")
    
    string_choice = input("Choose string case style [1-5] (default: 5): ").strip() or "5"
    string_case_style = string_styles.get(string_choice, StringCaseStyle.KEEP_ORIGINAL)
    
    cleaning_levels = {
        "1": TextCleaningLevel.MINIMAL,
        "2": TextCleaningLevel.MODERATE,
        "3": TextCleaningLevel.AGGRESSIVE
    }
    
    print("\nText cleaning level:")
    print("1. Minimal - Remove only extra spaces")
    print("2. Moderate - Remove spaces, basic special chars")
    print("3. Aggressive - Full cleaning, remove accents")
    
    cleaning_choice = input("Choose cleaning level [1-3] (default: 2): ").strip() or "2"
    text_cleaning_level = cleaning_levels.get(cleaning_choice, TextCleaningLevel.MODERATE)
    
    trim_spaces = input("Remove leading/trailing spaces? [Y/n]: ").strip().lower() != 'n'
    normalize_whitespace = input("Normalize whitespace (multiple spaces → single)? [Y/n]: ").strip().lower() != 'n'
    
    # Numeric formatting preferences
    print("\n🔢 NUMERIC FORMATTING PREFERENCES")
    print("-" * 36)
    numeric_formats = {
        "1": NumericFormat.DECIMAL_DOT,
        "2": NumericFormat.DECIMAL_COMMA,
        "3": NumericFormat.AUTO_DETECT
    }
    
    print("Decimal separator preference:")
    print("1. Dot (1234.56) - US/International standard")
    print("2. Comma (1234,56) - Brazilian/European standard")
    print("3. Auto-detect based on data")
    
    numeric_choice = input("Choose decimal format [1-3] (default: 1): ").strip() or "1"
    numeric_format = numeric_formats.get(numeric_choice, NumericFormat.DECIMAL_DOT)
    
    decimal_places_input = input("Standard decimal places (leave empty for auto): ").strip()
    decimal_places = int(decimal_places_input) if decimal_places_input.isdigit() else None
    
    remove_currency = input("Remove currency symbols (R$, $, €)? [Y/n]: ").strip().lower() != 'n'
    handle_thousands = input("Handle thousand separators (1,000 or 1.000)? [Y/n]: ").strip().lower() != 'n'
    
    # Date formatting preferences
    print("\n📅 DATE FORMATTING PREFERENCES")
    print("-" * 32)
    date_formats = {
        "1": DateFormat.ISO_FORMAT,
        "2": DateFormat.BR_FORMAT,
        "3": DateFormat.US_FORMAT,
        "4": DateFormat.AUTO_DETECT
    }
    
    print("Preferred date format:")
    print("1. ISO format (YYYY-MM-DD) - International standard")
    print("2. Brazilian format (DD/MM/YYYY)")
    print("3. US format (MM/DD/YYYY)")
    print("4. Auto-detect based on data")
    
    date_choice = input("Choose date format [1-4] (default: 1): ").strip() or "1"
    date_format = date_formats.get(date_choice, DateFormat.ISO_FORMAT)
    
    standardize_dates = input("Standardize all dates to chosen format? [Y/n]: ").strip().lower() != 'n'
    
    print("\nHandle invalid dates:")
    print("1. Drop rows with invalid dates")
    print("2. Fill with today's date") 
    print("3. Fill with custom date")
    invalid_date_choice = input("Choose option [1-3] (default: 1): ").strip() or "1"
    invalid_date_strategies = {"1": "drop", "2": "fill_today", "3": "fill_custom"}
    handle_invalid_dates = invalid_date_strategies.get(invalid_date_choice, "drop")
    
    # Missing value preferences
    print("\n❓ MISSING VALUE HANDLING")
    print("-" * 27)
    missing_strategies = {
        "1": MissingValueStrategy.DROP_ROWS,
        "2": MissingValueStrategy.FILL_ZERO,
        "3": MissingValueStrategy.FILL_MEAN,
        "4": MissingValueStrategy.FILL_MEDIAN,
        "5": MissingValueStrategy.FILL_MODE,
        "6": MissingValueStrategy.KEEP_AS_NULL
    }
    
    print("Strategy for missing NUMERIC values:")
    print("1. Drop rows with missing values")
    print("2. Fill with zero")
    print("3. Fill with mean")
    print("4. Fill with median (recommended)")
    print("5. Fill with mode")
    print("6. Keep as null")
    
    numeric_missing_choice = input("Choose strategy for NUMERIC missing values [1-6] (default: 4): ").strip() or "4"
    numeric_missing_strategy = missing_strategies.get(numeric_missing_choice, MissingValueStrategy.FILL_MEDIAN)
    
    print("\nStrategy for missing CATEGORICAL/TEXT values:")
    print("1. Drop rows with missing values")
    print("2. Fill with 'Unknown' or 'Not Specified'")
    print("3. Fill with most frequent value (mode)")
    print("4. Fill with 'Missing' label")
    print("5. Keep as null")
    
    categorical_strategies = {
        "1": MissingValueStrategy.DROP_ROWS,
        "2": MissingValueStrategy.FILL_CUSTOM,  # Will use "Unknown"
        "3": MissingValueStrategy.FILL_MODE,
        "4": MissingValueStrategy.FILL_CUSTOM,  # Will use "Missing"
        "5": MissingValueStrategy.KEEP_AS_NULL
    }
    
    categorical_missing_choice = input("Choose strategy for CATEGORICAL missing values [1-5] (default: 3): ").strip() or "3"
    categorical_missing_strategy = categorical_strategies.get(categorical_missing_choice, MissingValueStrategy.FILL_MODE)
    
    # Custom value for categorical if needed
    categorical_fill_value = "Unknown"
    if categorical_missing_choice in ["2", "4"]:
        if categorical_missing_choice == "2":
            categorical_fill_value = input("Enter custom value for missing categorical data (default: 'Unknown'): ").strip() or "Unknown"
        else:
            categorical_fill_value = input("Enter custom value for missing categorical data (default: 'Missing'): ").strip() or "Missing"
    
    print(f"\n📊 Column-level missing data handling:")
    print("When a column has too many missing values, should we remove the entire column?")
    missing_threshold_input = input("Drop columns with more than X% missing values (default: 70): ").strip() or "70"
    missing_threshold = float(missing_threshold_input) / 100.0
    
    # General preferences
    print("\n⚙️  GENERAL PREFERENCES")
    print("-" * 21)
    remove_duplicates = input("Automatically remove duplicate rows? [Y/n]: ").strip().lower() != 'n'
    handle_outliers = input("Detect and handle statistical outliers? [y/N]: ").strip().lower() == 'y'
    encoding_fix = input("Automatically fix encoding issues? [Y/n]: ").strip().lower() != 'n'
    memory_optimization = input("Optimize data types for memory efficiency? [Y/n]: ").strip().lower() != 'n'
    
    preferences = DataCleaningPreferences(
        column_naming_style=column_naming_style,
        remove_column_spaces=remove_spaces,
        remove_column_accents=remove_accents,
        remove_column_special_chars=remove_special,
        string_case_style=string_case_style,
        string_cleaning_level=text_cleaning_level,
        remove_leading_trailing_spaces=trim_spaces,
        normalize_whitespace=normalize_whitespace,
        numeric_format=numeric_format,
        decimal_places=decimal_places,
        remove_currency_symbols=remove_currency,
        handle_thousand_separators=handle_thousands,
        date_format=date_format,
        standardize_date_format=standardize_dates,
        handle_invalid_dates=handle_invalid_dates,
        numeric_missing_strategy=numeric_missing_strategy,
        categorical_missing_strategy=categorical_missing_strategy,
        categorical_fill_value=categorical_fill_value,
        missing_threshold=missing_threshold,
        remove_duplicates=remove_duplicates,
        handle_outliers=handle_outliers,
        encoding_fix=encoding_fix,
        memory_optimization=memory_optimization
    )
    
    print("\n✅ Configuration completed!")
    print("These preferences will be applied to your data cleaning pipeline.")
    
    return preferences

def save_preferences_to_file(preferences: DataCleaningPreferences, filepath: str = None):
    """
    Save user preferences to a JSON file for future use.
    
    Args:
        preferences: User's data cleaning preferences
        filepath: Path to save the preferences file (optional)
    """
    if filepath is None:
        # Create config directory if it doesn't exist
        import os
        config_dir = "config"
        os.makedirs(config_dir, exist_ok=True)
        filepath = os.path.join(config_dir, "data_cleaning_preferences.json")
    
    prefs_dict = {
        "column_naming_style": preferences.column_naming_style.value,
        "remove_column_spaces": preferences.remove_column_spaces,
        "remove_column_accents": preferences.remove_column_accents,
        "remove_column_special_chars": preferences.remove_column_special_chars,
        "string_case_style": preferences.string_case_style.value,
        "string_cleaning_level": preferences.string_cleaning_level.value,
        "remove_leading_trailing_spaces": preferences.remove_leading_trailing_spaces,
        "normalize_whitespace": preferences.normalize_whitespace,
        "numeric_format": preferences.numeric_format.value,
        "decimal_places": preferences.decimal_places,
        "remove_currency_symbols": preferences.remove_currency_symbols,
        "handle_thousand_separators": preferences.handle_thousand_separators,
        "date_format": preferences.date_format.value,
        "standardize_date_format": preferences.standardize_date_format,
        "handle_invalid_dates": preferences.handle_invalid_dates,
        "numeric_missing_strategy": preferences.numeric_missing_strategy.value,
        "categorical_missing_strategy": preferences.categorical_missing_strategy.value,
        "categorical_fill_value": preferences.categorical_fill_value,
        "missing_threshold": preferences.missing_threshold,
        "remove_duplicates": preferences.remove_duplicates,
        "handle_outliers": preferences.handle_outliers,
        "encoding_fix": preferences.encoding_fix,
        "memory_optimization": preferences.memory_optimization
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prefs_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Preferences saved to {filepath}")

def load_preferences_from_file(filepath: str = None) -> Optional[DataCleaningPreferences]:
    """
    Load user preferences from a JSON file.
    
    Args:
        filepath: Path to the preferences file (optional)
        
    Returns:
        DataCleaningPreferences object or None if file doesn't exist
    """
    if filepath is None:
        filepath = "config/data_cleaning_preferences.json"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prefs_dict = json.load(f)
        
        preferences = DataCleaningPreferences(
            column_naming_style=ColumnNamingStyle(prefs_dict["column_naming_style"]),
            remove_column_spaces=prefs_dict["remove_column_spaces"],
            remove_column_accents=prefs_dict["remove_column_accents"],
            remove_column_special_chars=prefs_dict["remove_column_special_chars"],
            string_case_style=StringCaseStyle(prefs_dict["string_case_style"]),
            string_cleaning_level=TextCleaningLevel(prefs_dict["string_cleaning_level"]),
            remove_leading_trailing_spaces=prefs_dict["remove_leading_trailing_spaces"],
            normalize_whitespace=prefs_dict["normalize_whitespace"],
            numeric_format=NumericFormat(prefs_dict["numeric_format"]),
            decimal_places=prefs_dict["decimal_places"],
            remove_currency_symbols=prefs_dict["remove_currency_symbols"],
            handle_thousand_separators=prefs_dict["handle_thousand_separators"],
            date_format=DateFormat(prefs_dict["date_format"]),
            standardize_date_format=prefs_dict["standardize_date_format"],
            handle_invalid_dates=prefs_dict["handle_invalid_dates"],
            numeric_missing_strategy=MissingValueStrategy(prefs_dict["numeric_missing_strategy"]),
            categorical_missing_strategy=MissingValueStrategy(prefs_dict["categorical_missing_strategy"]),
            categorical_fill_value=prefs_dict["categorical_fill_value"],
            missing_threshold=prefs_dict["missing_threshold"],
            remove_duplicates=prefs_dict["remove_duplicates"],
            handle_outliers=prefs_dict["handle_outliers"],
            encoding_fix=prefs_dict["encoding_fix"],
            memory_optimization=prefs_dict["memory_optimization"]
        )
        
        print(f"✅ Preferences loaded from {filepath}")
        return preferences
        
    except (FileNotFoundError, KeyError, ValueError) as e:
        # Silently return None instead of printing error messages
        return None

def get_user_preferences(use_saved: bool = True, force_interactive: bool = False) -> Optional[DataCleaningPreferences]:
    """
    Get user preferences, either from saved file or by collecting interactively.
    
    Args:
        use_saved: Whether to try loading from saved file first
        force_interactive: Force interactive collection even if saved file exists
        
    Returns:
        DataCleaningPreferences object or None if user skips
    """
    if use_saved and not force_interactive:
        saved_prefs = load_preferences_from_file()
        if saved_prefs:
            use_saved_choice = input(f"Found saved preferences. Use them? [Y/n]: ").strip().lower()
            if use_saved_choice != 'n':
                return saved_prefs
    
    # Ask if user wants to configure preferences
    if not force_interactive:
        config_choice = input("Configure custom data cleaning preferences? [y/N]: ").strip().lower()
        if config_choice != 'y':
            print("Skipping preference configuration. Using default patterns.")
            return None
    
    # Collect new preferences
    preferences = collect_user_preferences()
    
    # Ask if user wants to save (but don't force it)
    save_choice = input("\nSave these preferences for future use? [y/N]: ").strip().lower()
    if save_choice == 'y':
        save_preferences_to_file(preferences)
    else:
        print("Preferences will be used for this session only.")
    
    return preferences