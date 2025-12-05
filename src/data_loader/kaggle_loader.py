"""
Kaggle Loader - Loads processed job data into ChromaDB
Handles batch loading and progress tracking
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Optional
from tqdm import tqdm
from src.vector_store.chroma_manager import ChromaManager
from src.data_loader.data_processor import DataProcessor


class KaggleLoader:
    """
    Loads Kaggle job posting data into ChromaDB
    Handles batch processing for large datasets
    """
    
    def __init__(self, chroma_manager: ChromaManager = None):
        """
        Initialize Kaggle loader
        
        Args:
            chroma_manager: ChromaDB manager instance
        """
        self.chroma_manager = chroma_manager or ChromaManager()
        self.processor = DataProcessor()
    
    def load_jobs(
        self,
        max_jobs: Optional[int] = None,
        batch_size: int = 50,
        clear_existing: bool = False
    ) -> int:
        """
        Load jobs from Kaggle dataset into ChromaDB
        
        Args:
            max_jobs: Maximum number of jobs to load (None = all)
            batch_size: Number of jobs to load per batch
            clear_existing: Whether to clear existing jobs first
        
        Returns:
            Number of jobs loaded
        """
        print("\n" + "="*60)
        print("🚀 Loading Kaggle Jobs into ChromaDB")
        print("="*60 + "\n")
        
        # Clear existing if requested
        if clear_existing:
            print("🗑️  Clearing existing jobs...")
            self.chroma_manager.clear_jobs()
            print("   ✅ Existing jobs cleared\n")
        
        # Process jobs from CSV
        print("📊 Processing CSV files...")
        jobs = self.processor.process_jobs(max_jobs=max_jobs, clean=True)
        
        if not jobs:
            print("❌ No jobs to load!")
            return 0
        
        # Load in batches
        print(f"\n📥 Loading {len(jobs)} jobs into vector store...")
        print(f"   Batch size: {batch_size}")
        
        loaded_count = 0
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(jobs), batch_size), desc="Loading batches"):
            batch = jobs[i:i + batch_size]
            
            try:
                self.chroma_manager.add_jobs_batch(batch)
                loaded_count += len(batch)
            except Exception as e:
                print(f"\n⚠️  Error loading batch {i//batch_size + 1}: {e}")
                continue
        
        # Verify
        stats = self.chroma_manager.get_stats()
        
        print("\n" + "="*60)
        print("✅ Loading Complete!")
        print("="*60)
        print(f"📊 Total jobs in database: {stats['total_jobs']}")
        print(f"✅ Successfully loaded: {loaded_count} jobs")
        print("="*60 + "\n")
        
        return loaded_count
    
    def sample_and_load(
        self,
        sample_size: int = 500,
        filters: Optional[Dict] = None,
        clear_existing: bool = True
    ) -> int:
        """
        Load a sample of jobs (useful for testing/demo)
        
        Args:
            sample_size: Number of jobs to load
            filters: Optional filters (e.g., location, experience level)
            clear_existing: Whether to clear existing jobs
        
        Returns:
            Number of jobs loaded
        """
        print(f"\n🎯 Loading sample of {sample_size} jobs...")
        
        return self.load_jobs(
            max_jobs=sample_size,
            batch_size=50,
            clear_existing=clear_existing
        )
    
    def verify_data(self, n_samples: int = 3) -> None:
        """
        Verify loaded data by showing sample searches
        
        Args:
            n_samples: Number of sample searches to perform
        """
        print("\n" + "="*60)
        print("🔍 Verifying Loaded Data")
        print("="*60 + "\n")
        
        stats = self.chroma_manager.get_stats()
        print(f"📊 Total jobs in database: {stats['total_jobs']}\n")
        
        if stats['total_jobs'] == 0:
            print("❌ No jobs in database!")
            return
        
        # Sample queries
        sample_queries = [
            "Python developer with machine learning experience",
            "Data scientist with SQL and statistics background",
            "Frontend engineer React TypeScript"
        ]
        
        for i, query in enumerate(sample_queries[:n_samples], 1):
            print(f"Query {i}: '{query}'")
            
            results = self.chroma_manager.search_jobs(query, n_results=3)
            
            if results['documents']:
                for rank, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas']), 1):
                    print(f"\n   {rank}. {metadata.get('title', 'Unknown')} at {metadata.get('company', 'Unknown')}")
                    print(f"      Location: {metadata.get('location', 'Unknown')}")
                    print(f"      Preview: {doc[:100]}...")
            else:
                print("   No results found")
            
            print("\n" + "-"*60 + "\n")
        
        print("✅ Verification complete!\n")


# Testing and main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load Kaggle job data into ChromaDB")
    parser.add_argument("--max-jobs", type=int, default=None, help="Maximum number of jobs to load")
    parser.add_argument("--sample", type=int, default=None, help="Load only a sample (e.g., 500)")
    parser.add_argument("--clear", action="store_true", help="Clear existing jobs before loading")
    parser.add_argument("--verify", action="store_true", help="Verify data after loading")
    
    args = parser.parse_args()
    
    try:
        loader = KaggleLoader()
        
        if args.sample:
            # Load sample
            loader.sample_and_load(
                sample_size=args.sample,
                clear_existing=args.clear
            )
        else:
            # Load all (or max_jobs)
            loader.load_jobs(
                max_jobs=args.max_jobs,
                batch_size=50,
                clear_existing=args.clear
            )
        
        # Verify if requested
        if args.verify:
            loader.verify_data()
        
        print("✅ Job loading complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()