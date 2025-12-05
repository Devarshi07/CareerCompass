"""
Test script to verify the setup is working correctly.
Place this file in the project root directory.
"""

import sys
from pathlib import Path

# Ensure we're in the right directory
print(f"Current directory: {Path.cwd()}")
print(f"Python path: {sys.path}")

# Test imports
print("\n" + "="*60)
print("=== PHASE 1: Testing Basic Setup ===")
print("="*60)

try:
    from config.settings import settings
    print("✅ Config imported successfully")
except Exception as e:
    print(f"❌ Config import failed: {e}")
    sys.exit(1)

try:
    from src.llm.llm_factory import LLMFactory, MultiLLMRouter
    print("✅ LLM Factory imported successfully")
except Exception as e:
    print(f"❌ LLM Factory import failed: {e}")
    sys.exit(1)

# Test API keys
print("\n" + "="*60)
print("=== Testing API Keys ===")
print("="*60)
if settings.GROQ_API_KEY:
    print(f"✅ GROQ_API_KEY: {'*' * 20}{settings.GROQ_API_KEY[-4:]}")
else:
    print("⚠️  GROQ_API_KEY not set")

if settings.OPENAI_API_KEY:
    print(f"✅ OPENAI_API_KEY: {'*' * 20}{settings.OPENAI_API_KEY[-4:]}")
else:
    print("⚠️  OPENAI_API_KEY not set")

if settings.GOOGLE_API_KEY:
    print(f"✅ GOOGLE_API_KEY: {'*' * 20}{settings.GOOGLE_API_KEY[-4:]}")
else:
    print("⚠️  GOOGLE_API_KEY not set")

# Test LLM Factory
print("\n" + "="*60)
print("=== Testing LLM Factory ===")
print("="*60)
try:
    llm = LLMFactory()
    print(f"✅ LLM Factory initialized with provider: {llm.provider}")
    
    # Test generation
    print("\nTesting LLM Generation...")
    response = llm.generate(
        prompt="Say 'Hello, I am working!' in one sentence.",
        temperature=0.7,
        max_tokens=50
    )
    print(f"✅ Response: {response}")
    
except Exception as e:
    print(f"❌ LLM Factory test failed: {e}")
    import traceback
    traceback.print_exc()

# Test Phase 2: Embeddings
print("\n" + "="*60)
print("=== PHASE 2: Testing Embeddings ===")
print("="*60)

try:
    from src.embeddings.embedding_generator import EmbeddingGenerator
    print("✅ Embedding Generator imported successfully")
    
    generator = EmbeddingGenerator()
    print(f"✅ Embedding Generator initialized")
    print(f"   Model: {generator.model}")
    print(f"   Dimensions: {generator.dimensions}")
    
    # Test embedding generation
    test_text = "I am a software engineer with Python experience"
    print(f"\nGenerating embedding for: '{test_text}'")
    
    embedding = generator.generate_embedding(test_text)
    print(f"✅ Embedding generated!")
    print(f"   Dimension: {len(embedding)}")
    print(f"   First 5 values: {[round(x, 4) for x in embedding[:5]]}")
    
except Exception as e:
    print(f"❌ Embedding test failed: {e}")
    import traceback
    traceback.print_exc()

# Test Phase 2: ChromaDB
print("\n" + "="*60)
print("=== PHASE 2: Testing ChromaDB ===")
print("="*60)

try:
    from src.vector_store.chroma_manager import ChromaManager
    print("✅ ChromaManager imported successfully")
    
    manager = ChromaManager()
    print(f"✅ ChromaManager initialized")
    
    stats = manager.get_stats()
    print(f"   Jobs in store: {stats['total_jobs']}")
    print(f"   Resumes in store: {stats['total_resumes']}")
    
except Exception as e:
    print(f"❌ ChromaDB test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("=== All Tests Complete ===")
print("="*60)
print("\n✅ Your setup is ready for Phase 3!")