#!/usr/bin/env python3
"""
Setup script to help configure environment variables for the pandas pipeline agent.
"""

import os
import sys

def create_env_file():
    """Create a .env file with the required environment variables."""
    env_content = """# API Keys
GROQ_API_KEY=your_groq_api_key_here

# Other configuration
TEMPERATURE=0.1
MODEL_NAME=gpt-4o-mini
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file successfully!")
        print("\n📝 Next steps:")
        print("1. Edit the .env file and replace 'your_groq_api_key_here' with your actual Groq API key")
        print("2. You can get a free API key from: https://console.groq.com/")
        print("3. Run your application again")
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def check_env_setup():
    """Check if the environment is properly set up."""
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("❌ GROQ_API_KEY environment variable is not set")
        print("\n🔧 To fix this, you can:")
        print("1. Create a .env file (run: python setup_env.py)")
        print("2. Set the environment variable directly: export GROQ_API_KEY=your_key")
        print("3. Pass the API key directly in your code")
        return False
    elif api_key == "your_groq_api_key_here":
        print("❌ GROQ_API_KEY is set but still has the placeholder value")
        print("Please edit your .env file and replace 'your_groq_api_key_here' with your actual API key")
        return False
    else:
        print("✅ GROQ_API_KEY is properly configured")
        return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        create_env_file()
    else:
        check_env_setup() 