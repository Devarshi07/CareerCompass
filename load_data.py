"""
Main script to load Kaggle job data into ChromaDB
Run this once to populate your vector database

Usage:
    python load_data.py                    # Load sample (500 jobs)
    python load_data.py --all              # Load all jobs
    python load_data.py --sample 1000      # Load 1000 jobs
    python load_data.py --clear            # Clear existing data first
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data_loader.kaggle_loader import KaggleLoader


def main():
    """Main function to load data"""
    
    print("\n" + "="*70)
    print("🚀 JOB SEARCH RAG ASSISTANT - DATA LOADER")
    print("="*70 + "\n")
    
    print("This script will load job postings from Kaggle into ChromaDB")
    print("for semantic search and matching.\n")
    
    # Get user input
    print("Options:")
    print("  1. Load sample (500 jobs) - RECOMMENDED for testing")
    print("  2. Load more (1000 jobs) - Good for demo")
    print("  3. Load many (5000 jobs) - Full experience")
    print("  4. Load ALL jobs (~33k) - Takes time\n")
    
    choice = input("Enter your choice (1-4) [default: 1]: ").strip() or "1"
    
    # Map choice to job count
    job_counts = {
        "1": 500,
        "2": 1000,
        "3": 5000,
        "4": None  # None = all jobs
    }
    
    max_jobs = job_counts.get(choice, 500)
    
    # Ask about clearing existing data
    clear = input("\nClear existing jobs in database? (y/n) [default: y]: ").strip().lower()
    clear_existing = clear != 'n'
    
    print("\n" + "-"*70 + "\n")
    
    try:
        # Initialize loader
        loader = KaggleLoader()
        
        # Load data
        if max_jobs:
            print(f"📥 Loading up to {max_jobs} jobs...\n")
            loader.load_jobs(
                max_jobs=max_jobs,
                batch_size=50,
                clear_existing=clear_existing
            )
        else:
            print(f"📥 Loading ALL jobs (this may take a while)...\n")
            loader.load_jobs(
                max_jobs=None,
                batch_size=100,
                clear_existing=clear_existing
            )
        
        # Verify data
        print("\n🔍 Running verification...\n")
        loader.verify_data(n_samples=3)
        
        print("="*70)
        print("✅ DATA LOADING COMPLETE!")
        print("="*70)
        print("\n💡 Next steps:")
        print("   - Test with: python test_setup.py")
        print("   - Run app with: streamlit run app/streamlit_app.py")
        print()
        
    except FileNotFoundError as e:
        print("\n❌ ERROR: CSV files not found!")
        print(f"   {e}")
        print("\n📝 Make sure your Kaggle CSV files are in: data/kaggle/")
        print("   Required files:")
        print("   - postings.csv")
        print("   - companies.csv (optional)")
        print("   - skills.csv (optional)")
        print("   - job_skills.csv (optional)")
        print()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()