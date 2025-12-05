"""
ChromaDB Manager - Handles vector storage and retrieval
Manages job descriptions and resume collections with semantic search
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import uuid

from config.settings import settings
from src.embeddings.embedding_generator import EmbeddingGenerator


class ChromaManager:
    """
    Manages ChromaDB collections for jobs and resumes.
    Provides semantic search and document management.
    """
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize ChromaDB manager
        
        Args:
            persist_directory: Directory to persist the vector store
        """
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        # Get or create collections
        self.jobs_collection = self._get_or_create_collection(
            settings.CHROMA_COLLECTION_JOBS
        )
        self.resumes_collection = self._get_or_create_collection(
            settings.CHROMA_COLLECTION_RESUMES
        )
        
        print(f"✅ ChromaDB initialized at: {self.persist_directory}")
        print(f"   Jobs collection: {self.jobs_collection.count()} documents")
        print(f"   Resumes collection: {self.resumes_collection.count()} documents")
    
    def _get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection"""
        try:
            collection = self.client.get_collection(name=collection_name)
            print(f"   Loaded existing collection: {collection_name}")
        except:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            print(f"   Created new collection: {collection_name}")
        
        return collection
    
    def add_job(
        self, 
        job_description: str, 
        metadata: Dict[str, Any],
        job_id: str = None
    ) -> str:
        """
        Add a job description to the vector store
        
        Args:
            job_description: Full text of the job description
            metadata: Additional metadata (title, company, location, etc.)
            job_id: Optional custom job ID
        
        Returns:
            ID of the added job
        """
        if not job_id:
            job_id = f"job_{uuid.uuid4()}"
        
        # Generate embedding
        embedding = self.embedding_generator.generate_embedding(job_description)
        
        # Add to collection
        self.jobs_collection.add(
            documents=[job_description],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[job_id]
        )
        
        print(f"✅ Added job: {metadata.get('title', 'Unknown')} (ID: {job_id})")
        return job_id
    
    def add_jobs_batch(
        self, 
        jobs: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add multiple jobs in batch
        
        Args:
            jobs: List of dicts with 'description' and 'metadata' keys
        
        Returns:
            List of job IDs
        """
        documents = []
        metadatas = []
        ids = []
        
        for job in jobs:
            job_id = job.get('id', f"job_{uuid.uuid4()}")
            documents.append(job['description'])
            metadatas.append(job['metadata'])
            ids.append(job_id)
        
        # Generate embeddings in batch
        embeddings = self.embedding_generator.generate_embeddings(documents)
        
        # Add to collection
        self.jobs_collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added {len(jobs)} jobs to collection")
        return ids
    
    def add_resume(
        self,
        resume_text: str,
        metadata: Dict[str, Any],
        resume_id: str = None
    ) -> str:
        """
        Add a resume to the vector store
        
        Args:
            resume_text: Full text of the resume
            metadata: Additional metadata (name, email, etc.)
            resume_id: Optional custom resume ID
        
        Returns:
            ID of the added resume
        """
        if not resume_id:
            resume_id = f"resume_{uuid.uuid4()}"
        
        # Generate embedding
        embedding = self.embedding_generator.generate_embedding(resume_text)
        
        # Add to collection
        self.resumes_collection.add(
            documents=[resume_text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[resume_id]
        )
        
        print(f"✅ Added resume (ID: {resume_id})")
        return resume_id
    
    def search_jobs(
        self,
        query: str,
        n_results: int = None,
        where_filter: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant jobs using semantic search
        
        Args:
            query: Search query (e.g., resume text or skills)
            n_results: Number of results to return
            where_filter: Optional metadata filter
        
        Returns:
            Dictionary with documents, metadatas, distances, and ids
        """
        n_results = n_results or settings.TOP_K_RETRIEVAL
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search
        results = self.jobs_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        return self._format_results(results)
    
    def search_resumes(
        self,
        query: str,
        n_results: int = None,
        where_filter: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant resumes using semantic search
        
        Args:
            query: Search query (e.g., job description or required skills)
            n_results: Number of results to return
            where_filter: Optional metadata filter
        
        Returns:
            Dictionary with documents, metadatas, distances, and ids
        """
        n_results = n_results or settings.TOP_K_RETRIEVAL
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search
        results = self.resumes_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        return self._format_results(results)
    
    def _format_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format ChromaDB results for easier use"""
        if not results['ids'] or not results['ids'][0]:
            return {
                'documents': [],
                'metadatas': [],
                'distances': [],
                'ids': []
            }
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0],
            'ids': results['ids'][0]
        }
    
    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific job by ID"""
        try:
            result = self.jobs_collection.get(ids=[job_id])
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except:
            pass
        return None
    
    def get_resume_by_id(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific resume by ID"""
        try:
            result = self.resumes_collection.get(ids=[resume_id])
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except:
            pass
        return None
    
    def delete_job(self, job_id: str):
        """Delete a job from the collection"""
        self.jobs_collection.delete(ids=[job_id])
        print(f"✅ Deleted job: {job_id}")
    
    def delete_resume(self, resume_id: str):
        """Delete a resume from the collection"""
        self.resumes_collection.delete(ids=[resume_id])
        print(f"✅ Deleted resume: {resume_id}")
    
    def clear_jobs(self):
        """Clear all jobs from the collection"""
        self.client.delete_collection(settings.CHROMA_COLLECTION_JOBS)
        self.jobs_collection = self._get_or_create_collection(
            settings.CHROMA_COLLECTION_JOBS
        )
        print("✅ Cleared all jobs")
    
    def clear_resumes(self):
        """Clear all resumes from the collection"""
        self.client.delete_collection(settings.CHROMA_COLLECTION_RESUMES)
        self.resumes_collection = self._get_or_create_collection(
            settings.CHROMA_COLLECTION_RESUMES
        )
        print("✅ Cleared all resumes")
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the collections"""
        return {
            'total_jobs': self.jobs_collection.count(),
            'total_resumes': self.resumes_collection.count()
        }


# Testing
if __name__ == "__main__":
    print("=== Testing ChromaDB Manager ===\n")
    
    try:
        # Initialize manager
        manager = ChromaManager()
        
        # Add sample jobs
        sample_jobs = [
            {
                'description': "We are looking for a Senior Python Developer with 5+ years of experience in Django and Flask. Must have strong knowledge of REST APIs and PostgreSQL.",
                'metadata': {
                    'title': 'Senior Python Developer',
                    'company': 'TechCorp',
                    'location': 'San Francisco, CA'
                }
            },
            {
                'description': "Seeking a Data Scientist with expertise in machine learning, NLP, and deep learning. Experience with PyTorch and TensorFlow required.",
                'metadata': {
                    'title': 'Data Scientist',
                    'company': 'AI Solutions',
                    'location': 'Remote'
                }
            }
        ]
        
        print("Adding sample jobs...")
        manager.add_jobs_batch(sample_jobs)
        
        # Test search
        query = "I am a Python developer with experience in web frameworks and databases"
        print(f"\n\nSearching for: '{query}'\n")
        
        results = manager.search_jobs(query, n_results=2)
        
        print("✅ Search Results:")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'], 
            results['metadatas'], 
            results['distances']
        )):
            print(f"\n{i+1}. {metadata['title']} at {metadata['company']}")
            print(f"   Similarity: {1 - distance:.4f}")
            print(f"   Description: {doc[:100]}...")
        
        # Get stats
        print("\n\n" + "="*50)
        stats = manager.get_stats()
        print(f"Total Jobs: {stats['total_jobs']}")
        print(f"Total Resumes: {stats['total_resumes']}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()