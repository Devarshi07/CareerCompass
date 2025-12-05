"""
Configuration settings for the Job Search RAG Assistant.
Loads environment variables and defines model configurations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Get project root directory (parent of config directory)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

class Settings:
    """Application settings and configuration"""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL = "openai"  # Using OpenAI for embeddings
    PRIMARY_LLM = os.getenv("PRIMARY_LLM", "groq")  # groq, openai, gemini
    
    # LLM Model Names
    GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast, powerful, free
    OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective OpenAI model
    GEMINI_MODEL = "models/gemini-2.5-flash"  # Gemini model (needs 'models/' prefix)
    
    # Embedding Model Names
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, cost-effective
    
    # Vector Store Configuration
    CHROMA_PERSIST_DIR = os.path.join("data", "vector_store")
    CHROMA_COLLECTION_JOBS = "job_descriptions"
    CHROMA_COLLECTION_RESUMES = "user_resumes"
    
    # RAG Configuration
    TOP_K_RETRIEVAL = 3  # Number of relevant chunks to retrieve
    CHUNK_SIZE = 500  # Characters per chunk
    CHUNK_OVERLAP = 50  # Overlap between chunks
    
    # Agent Configuration
    AGENT_TEMPERATURE = 0.7  # Creativity vs consistency
    MAX_TOKENS = 2000  # Maximum response length
    MIN_JOB_MATCH_SCORE = 0.60  # Minimum overall match score (60%) for jobs to be included
    
    # File Upload Configuration
    ALLOWED_RESUME_TYPES = ["pdf", "docx", "txt"]
    MAX_FILE_SIZE_MB = 5
    
    # Data Directories
    SAMPLE_JOBS_DIR = BASE_DIR / "data" / "sample_jobs"
    SAMPLE_RESUMES_DIR = BASE_DIR / "data" / "sample_resumes"
    DATA_DIR = BASE_DIR / "data"
    
    @classmethod
    def validate_api_keys(cls):
        """Validate that required API keys are present"""
        errors = []
        
        # Always need OpenAI for embeddings
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required for embeddings")
        
        # Check LLM provider
        if cls.PRIMARY_LLM == "groq" and not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required when PRIMARY_LLM is 'groq'")
        
        if cls.PRIMARY_LLM == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when PRIMARY_LLM is 'openai'")
        
        if cls.PRIMARY_LLM == "gemini" and not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is required when PRIMARY_LLM is 'gemini'")
        
        if errors:
            raise ValueError("\n".join(errors))
        
        return True
    
    @classmethod
    def get_embedding_dimensions(cls):
        """Return embedding dimensions for OpenAI"""
        return 1536  # text-embedding-3-small dimensions
    
    @classmethod
    def create_data_directories(cls):
        """Create necessary data directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.SAMPLE_JOBS_DIR,
            cls.SAMPLE_RESUMES_DIR,
            Path(cls.CHROMA_PERSIST_DIR)
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ Data directories created successfully")


# Create a global settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate_api_keys()
    settings.create_data_directories()
except ValueError as e:
    print(f"⚠️  Configuration Error: {e}")
    print("Please check your .env file and ensure all required API keys are set.")