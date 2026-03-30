"""
API configuration module for managing API keys and client initialization
"""
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()


class APIConfig:
    """Manages API keys and client configurations"""
    
    @staticmethod
    def get_openai_api_key() -> Optional[str]:
        """Get OpenAI API key from environment variables"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        return api_key
    
    @staticmethod
    def get_google_api_key() -> Optional[str]:
        """Get Google API key from environment variables"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        return api_key
    
    @staticmethod
    def validate_api_keys() -> dict:
        """
        Validate that all required API keys are present
        
        Returns:
            Dictionary with validation status for each API
        """
        results = {
            'openai': False,
            'google': False
        }
        
        try:
            APIConfig.get_openai_api_key()
            results['openai'] = True
        except ValueError:
            pass
        
        try:
            APIConfig.get_google_api_key()
            results['google'] = True
        except ValueError:
            pass
        
        return results
