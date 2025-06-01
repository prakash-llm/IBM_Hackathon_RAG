import os
from dotenv import load_dotenv
from loguru import logger

def load_config():
    """Load configuration from environment variables and .env file"""
    load_dotenv()
    
    config = {
        # IBM watsonx credentials
        'ibm_api_key': os.getenv('IBM_API_KEY'),
        'ibm_project_id': os.getenv('IBM_PROJECT_ID'),
        'ibm_url': os.getenv('IBM_URL'),
        
        # Vector store connection IDs
        'employer_connection_id': os.getenv('EMPLOYER_CONNECTION_ID', 'e21ce8ff-b29a-481c-81d9-d0102c321d0f'),
        'member_connection_id': os.getenv('MEMBER_CONNECTION_ID', '26702b6c-4b53-4964-b32e-b8e4ba851ca1'),
        
        # Model configuration
        'model_id': os.getenv('MODEL_ID', 'watsonx/meta-llama/llama-3-3-70b-instruct'),
        'embedding_model_id': os.getenv('EMBEDDING_MODEL_ID', 'ibm/granite-embedding-107m-multilingual'),
        'temperature': float(os.getenv('TEMPERATURE', '0.7')),
        'max_tokens': int(os.getenv('MAX_TOKENS', '1000'))
    }
    
    return config 