from langchain_groq import ChatGroq
import os

def get_llm(model_name: str, temperature: float = 0.1, api_key: str = None):
    """
    Get an LLM instance with proper error handling for missing API keys.
    
    Args:
        model_name: The name of the model to use
        temperature: The temperature for generation (default: 0.1)
        api_key: The API key to use (if None, will try to get from environment)
    
    Returns:
        A ChatGroq instance
        
    Raises:
        ValueError: If no API key is provided and GROQ_API_KEY is not set in environment
    """
    # Try to get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is required but not set. Please:\n"
            "1. Set the GROQ_API_KEY environment variable, or\n"
            "2. Create a .env file with: GROQ_API_KEY=your_actual_api_key_here, or\n"
            "3. Pass the api_key parameter directly to get_llm()\n"
            "You can get a free API key from: https://console.groq.com/"
        )
    
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )
