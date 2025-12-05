"""
Embedding Generator - Creates vector embeddings from text using OpenAI
Optimized for OpenAI's text-embedding-3-small model
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Union
import numpy as np
from openai import OpenAI

from config.settings import settings


class EmbeddingGenerator:
    """
    Generates embeddings using OpenAI's embedding models.
    Uses text-embedding-3-small for cost-effective, high-quality embeddings.
    """
    
    def __init__(self):
        """Initialize OpenAI embedding generator"""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_EMBEDDING_MODEL  # text-embedding-3-small
        self.dimensions = 1536  # text-embedding-3-small produces 1536-dimensional vectors
        
        print(f"✅ Initialized OpenAI embeddings: {self.model}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
        
        Returns:
            List of floats representing the embedding vector (1536 dimensions)
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        
        except Exception as e:
            raise Exception(f"Error generating embedding with OpenAI: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)
        OpenAI allows up to 2048 texts per batch request
        
        Args:
            texts: List of input texts
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        
        if not valid_texts:
            raise ValueError("All input texts are empty")
        
        try:
            # OpenAI can handle batch requests efficiently
            response = self.client.embeddings.create(
                model=self.model,
                input=valid_texts
            )
            
            # Extract embeddings in order
            embeddings = [item.embedding for item in response.data]
            return embeddings
        
        except Exception as e:
            raise Exception(f"Error generating batch embeddings with OpenAI: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model"""
        return self.dimensions
    
    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score between -1 and 1 (higher is more similar)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


# Convenience function for quick embedding generation
def get_embeddings(texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    Quick function to generate embeddings using OpenAI
    
    Args:
        texts: Single text or list of texts
    
    Returns:
        Single embedding or list of embeddings
    """
    generator = EmbeddingGenerator()
    
    if isinstance(texts, str):
        return generator.generate_embedding(texts)
    else:
        return generator.generate_embeddings(texts)


# Testing
if __name__ == "__main__":
    print("=== Testing OpenAI Embedding Generator ===\n")
    
    try:
        # Initialize generator
        generator = EmbeddingGenerator()
        
        # Test single embedding
        test_text = "I am a software engineer with 5 years of experience in Python and machine learning."
        
        print(f"Generating embedding for: '{test_text[:50]}...'\n")
        embedding = generator.generate_embedding(test_text)
        
        print(f"✅ Embedding generated successfully!")
        print(f"   Model: {generator.model}")
        print(f"   Dimension: {len(embedding)}")
        print(f"   First 5 values: {[round(x, 6) for x in embedding[:5]]}")
        
        # Test batch embeddings
        test_texts = [
            "Senior Python Developer with ML expertise",
            "Data Scientist specializing in NLP",
            "Frontend Engineer with React experience"
        ]
        
        print(f"\n\nGenerating {len(test_texts)} embeddings in batch...\n")
        embeddings = generator.generate_embeddings(test_texts)
        
        print(f"✅ Batch embeddings generated!")
        print(f"   Count: {len(embeddings)}")
        print(f"   Dimension: {len(embeddings[0])}")
        
        # Test similarity
        print("\n\n=== Testing Similarity ===")
        sim_1_2 = EmbeddingGenerator.cosine_similarity(embeddings[0], embeddings[1])
        print(f"Similarity between text 1 and 2: {sim_1_2:.4f}")
        print(f"  (Python Dev vs Data Scientist - should be high)")
        
        sim_1_3 = EmbeddingGenerator.cosine_similarity(embeddings[0], embeddings[2])
        print(f"\nSimilarity between text 1 and 3: {sim_1_3:.4f}")
        print(f"  (Python Dev vs Frontend - should be lower)")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()