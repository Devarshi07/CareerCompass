"""
RAG Retriever - Retrieve relevant documents from vector store
Handles semantic search and result ranking
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Any, Optional
from src.vector_store.chroma_manager import ChromaManager
from config.settings import settings


class Retriever:
    """
    Retrieves relevant documents from the vector store using semantic search
    """
    
    def __init__(self, chroma_manager: ChromaManager = None):
        """
        Initialize retriever
        
        Args:
            chroma_manager: ChromaDB manager instance (creates new if None)
        """
        self.chroma_manager = chroma_manager or ChromaManager()
    
    def retrieve_jobs(
        self,
        query: str,
        n_results: int = None,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant job descriptions
        
        Args:
            query: Search query (e.g., resume text or skills)
            n_results: Number of results to return
            filters: Optional metadata filters (e.g., {'location': 'Remote'})
        
        Returns:
            List of job dictionaries with content and metadata
        """
        n_results = n_results or settings.TOP_K_RETRIEVAL
        
        # Search vector store
        results = self.chroma_manager.search_jobs(
            query=query,
            n_results=n_results,
            where_filter=filters
        )
        
        # Format results
        jobs = []
        for i, (doc, metadata, distance, doc_id) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances'],
            results['ids']
        )):
            jobs.append({
                'rank': i + 1,
                'id': doc_id,
                'content': doc,
                'metadata': metadata,
                'similarity_score': 1 - distance,  # Convert distance to similarity
                'distance': distance
            })
        
        return jobs
    
    def retrieve_resumes(
        self,
        query: str,
        n_results: int = None,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant resumes
        
        Args:
            query: Search query (e.g., job description or required skills)
            n_results: Number of results to return
            filters: Optional metadata filters
        
        Returns:
            List of resume dictionaries with content and metadata
        """
        n_results = n_results or settings.TOP_K_RETRIEVAL
        
        # Search vector store
        results = self.chroma_manager.search_resumes(
            query=query,
            n_results=n_results,
            where_filter=filters
        )
        
        # Format results
        resumes = []
        for i, (doc, metadata, distance, doc_id) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances'],
            results['ids']
        )):
            resumes.append({
                'rank': i + 1,
                'id': doc_id,
                'content': doc,
                'metadata': metadata,
                'similarity_score': 1 - distance,
                'distance': distance
            })
        
        return resumes
    
    def retrieve_with_context(
        self,
        query: str,
        collection_type: str = 'jobs',
        n_results: int = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve documents and format them for LLM context
        
        Args:
            query: Search query
            collection_type: 'jobs' or 'resumes'
            n_results: Number of results
            include_metadata: Whether to include metadata in context
        
        Returns:
            Dictionary with formatted context and results
        """
        if collection_type == 'jobs':
            results = self.retrieve_jobs(query, n_results)
        elif collection_type == 'resumes':
            results = self.retrieve_resumes(query, n_results)
        else:
            raise ValueError(f"Invalid collection_type: {collection_type}")
        
        # Format context string
        context_parts = []
        for result in results:
            context_parts.append(f"[Document {result['rank']}]")
            
            if include_metadata and result['metadata']:
                # Add relevant metadata
                metadata_str = ", ".join([
                    f"{k}: {v}" for k, v in result['metadata'].items()
                ])
                context_parts.append(f"Metadata: {metadata_str}")
            
            context_parts.append(f"Content: {result['content']}")
            context_parts.append(f"Similarity: {result['similarity_score']:.3f}")
            context_parts.append("")  # Empty line between documents
        
        return {
            'context': "\n".join(context_parts),
            'results': results,
            'num_results': len(results)
        }
    
    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific job by ID"""
        return self.chroma_manager.get_job_by_id(job_id)
    
    def get_resume_by_id(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific resume by ID"""
        return self.chroma_manager.get_resume_by_id(resume_id)
    
    def rerank_results(
        self,
        results: List[Dict[str, Any]],
        boost_keywords: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results based on additional criteria
        
        Args:
            results: List of retrieved results
            boost_keywords: Keywords to boost in ranking
        
        Returns:
            Re-ranked results
        """
        if not boost_keywords:
            return results
        
        # Calculate boost score
        for result in results:
            boost_score = 0
            content_lower = result['content'].lower()
            
            for keyword in boost_keywords:
                if keyword.lower() in content_lower:
                    boost_score += 1
            
            # Combine similarity score with boost
            result['boosted_score'] = result['similarity_score'] + (boost_score * 0.1)
        
        # Sort by boosted score
        results.sort(key=lambda x: x.get('boosted_score', x['similarity_score']), reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results


# Testing
if __name__ == "__main__":
    print("=== Testing Retriever ===\n")
    
    try:
        from src.vector_store.chroma_manager import ChromaManager
        
        # Initialize
        manager = ChromaManager()
        retriever = Retriever(chroma_manager=manager)
        
        # Add sample data if collections are empty
        if manager.get_stats()['total_jobs'] == 0:
            print("Adding sample jobs for testing...")
            sample_jobs = [
                {
                    'description': "We're seeking a Senior Python Developer with 5+ years of experience. Must have expertise in Django, Flask, and PostgreSQL. Experience with AWS and Docker is required.",
                    'metadata': {
                        'title': 'Senior Python Developer',
                        'company': 'TechCorp',
                        'location': 'San Francisco, CA',
                        'salary': '$120k-$150k'
                    }
                },
                {
                    'description': "Data Scientist position requiring strong skills in machine learning, NLP, and deep learning. PyTorch and TensorFlow experience mandatory. PhD preferred.",
                    'metadata': {
                        'title': 'Data Scientist',
                        'company': 'AI Solutions',
                        'location': 'Remote',
                        'salary': '$130k-$160k'
                    }
                },
                {
                    'description': "Frontend Engineer needed with React, TypeScript, and modern web development skills. Experience with responsive design and REST APIs.",
                    'metadata': {
                        'title': 'Frontend Engineer',
                        'company': 'WebCo',
                        'location': 'New York, NY',
                        'salary': '$100k-$130k'
                    }
                }
            ]
            
            manager.add_jobs_batch(sample_jobs)
            print("✅ Sample jobs added\n")
        
        # Test retrieval
        query = "I am a Python developer with experience in web frameworks and cloud technologies"
        print(f"Query: '{query}'\n")
        
        results = retriever.retrieve_jobs(query, n_results=3)
        
        print(f"✅ Retrieved {len(results)} results:\n")
        for result in results:
            print(f"{result['rank']}. {result['metadata']['title']} at {result['metadata']['company']}")
            print(f"   Similarity: {result['similarity_score']:.3f}")
            print(f"   Content preview: {result['content'][:80]}...")
            print()
        
        # Test context formatting
        print("\n=== Testing Context Formatting ===\n")
        context_data = retriever.retrieve_with_context(query, n_results=2)
        print(f"✅ Context created with {context_data['num_results']} results")
        print(f"\nContext preview:\n{context_data['context'][:300]}...\n")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()