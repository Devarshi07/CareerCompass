"""
Data Processor - Reads and combines Kaggle LinkedIn job posting CSVs
Joins multiple tables to create rich job descriptions
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from typing import Dict, List, Optional
from config.settings import settings


class DataProcessor:
    """
    Processes Kaggle LinkedIn job posting data
    Combines multiple CSV files into rich job descriptions
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize data processor
        
        Args:
            data_dir: Directory containing CSV files (default: data/kaggle/)
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Use project root / data / kaggle
            project_root = Path(__file__).resolve().parent.parent.parent
            self.data_dir = project_root / "data" / "kaggle"
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        print(f"📂 Loading data from: {self.data_dir}")
    
    def load_postings(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load job postings CSV
        
        Args:
            max_rows: Maximum number of rows to load (for testing)
        
        Returns:
            DataFrame with job postings
        """
        postings_path = self.data_dir / "postings.csv"
        
        if not postings_path.exists():
            raise FileNotFoundError(f"postings.csv not found at {postings_path}")
        
        print(f"📄 Loading postings.csv...")
        df = pd.read_csv(postings_path, nrows=max_rows)
        print(f"   ✅ Loaded {len(df)} job postings")
        
        return df
    
    def load_companies(self) -> pd.DataFrame:
        """Load companies CSV"""
        companies_path = self.data_dir / "companies.csv"
        
        if not companies_path.exists():
            print("   ⚠️  companies.csv not found, skipping")
            return pd.DataFrame()
        
        print(f"📄 Loading companies.csv...")
        df = pd.read_csv(companies_path)
        print(f"   ✅ Loaded {len(df)} companies")
        
        return df
    
    def load_skills(self) -> pd.DataFrame:
        """Load skills mapping"""
        skills_path = self.data_dir / "skills.csv"
        
        if not skills_path.exists():
            print("   ⚠️  skills.csv not found, skipping")
            return pd.DataFrame()
        
        print(f"📄 Loading skills.csv...")
        df = pd.read_csv(skills_path)
        print(f"   ✅ Loaded {len(df)} skills")
        
        return df
    
    def load_job_skills(self) -> pd.DataFrame:
        """Load job-skills relationship"""
        job_skills_path = self.data_dir / "job_skills.csv"
        
        if not job_skills_path.exists():
            print("   ⚠️  job_skills.csv not found, skipping")
            return pd.DataFrame()
        
        print(f"📄 Loading job_skills.csv...")
        df = pd.read_csv(job_skills_path)
        print(f"   ✅ Loaded {len(df)} job-skill relationships")
        
        return df
    
    def load_salaries(self) -> pd.DataFrame:
        """Load salary information"""
        salaries_path = self.data_dir / "salaries.csv"
        
        if not salaries_path.exists():
            print("   ⚠️  salaries.csv not found, skipping")
            return pd.DataFrame()
        
        print(f"📄 Loading salaries.csv...")
        df = pd.read_csv(salaries_path)
        print(f"   ✅ Loaded {len(df)} salary records")
        
        return df
    
    def combine_job_skills(self, postings_df: pd.DataFrame, job_skills_df: pd.DataFrame, skills_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine job postings with their required skills
        
        Args:
            postings_df: Job postings
            job_skills_df: Job-skill relationships
            skills_df: Skill names
        
        Returns:
            DataFrame with skills added
        """
        if job_skills_df.empty or skills_df.empty:
            print("   ⚠️  Skipping skills joining (missing data)")
            postings_df['required_skills'] = ""
            return postings_df
        
        print("🔗 Joining job skills...")
        
        # Join job_skills with skills to get skill names
        job_skills_with_names = job_skills_df.merge(
            skills_df, 
            on='skill_abr', 
            how='left'
        )
        
        # Group skills by job_id
        skills_grouped = job_skills_with_names.groupby('job_id')['skill_name'].apply(
            lambda x: ', '.join(x.dropna().astype(str))
        ).reset_index()
        skills_grouped.columns = ['job_id', 'required_skills']
        
        # Merge with postings
        postings_with_skills = postings_df.merge(
            skills_grouped,
            on='job_id',
            how='left'
        )
        
        # Fill NaN with empty string
        postings_with_skills['required_skills'] = postings_with_skills['required_skills'].fillna('')
        
        print(f"   ✅ Added skills to {len(postings_with_skills)} postings")
        
        return postings_with_skills
    
    def process_jobs(self, max_jobs: Optional[int] = None, clean: bool = True) -> List[Dict]:
        """
        Process all job data and create rich job documents
        
        Args:
            max_jobs: Maximum number of jobs to process
            clean: Whether to clean/filter data
        
        Returns:
            List of processed job dictionaries
        """
        print("\n" + "="*60)
        print("🚀 Processing Kaggle Job Data")
        print("="*60 + "\n")
        
        # Load all data
        postings_df = self.load_postings(max_rows=max_jobs)
        companies_df = self.load_companies()
        skills_df = self.load_skills()
        job_skills_df = self.load_job_skills()
        salaries_df = self.load_salaries()
        
        # Combine with skills
        if not job_skills_df.empty and not skills_df.empty:
            postings_df = self.combine_job_skills(postings_df, job_skills_df, skills_df)
        else:
            postings_df['required_skills'] = postings_df.get('skills_desc', '')
        
        # Merge with companies if available
        if not companies_df.empty and 'company_id' in postings_df.columns:
            print("🔗 Joining company information...")
            postings_df = postings_df.merge(
                companies_df[['company_id', 'name', 'description']],
                on='company_id',
                how='left',
                suffixes=('', '_company')
            )
            print("   ✅ Company info added")
        
        # Clean data if requested
        if clean:
            print("\n🧹 Cleaning data...")
            initial_count = len(postings_df)
            
            # Remove jobs without descriptions
            postings_df = postings_df[postings_df['description'].notna()]
            postings_df = postings_df[postings_df['description'].str.len() > 50]
            
            # Remove duplicates
            postings_df = postings_df.drop_duplicates(subset=['title', 'description'])
            
            print(f"   ✅ Kept {len(postings_df)} jobs (removed {initial_count - len(postings_df)})")
        
        # Convert to list of dictionaries
        print("\n📦 Creating job documents...")
        jobs = []
        
        for idx, row in postings_df.iterrows():
            # Build rich job description
            job_text = self._build_job_text(row)
            
            # Build metadata
            metadata = self._build_metadata(row)
            
            jobs.append({
                'id': f"job_{row['job_id']}",
                'description': job_text,
                'metadata': metadata
            })
        
        print(f"   ✅ Created {len(jobs)} job documents")
        print("\n" + "="*60)
        print(f"✅ Processing complete: {len(jobs)} jobs ready")
        print("="*60 + "\n")
        
        return jobs
    
    def _build_job_text(self, row: pd.Series) -> str:
        """Build rich job description text"""
        parts = []
        
        # Title
        if pd.notna(row.get('title')):
            parts.append(f"Job Title: {row['title']}")
        
        # Company
        company_name = row.get('company_name') or row.get('name_company')
        if pd.notna(company_name):
            parts.append(f"Company: {company_name}")
        
        # Location
        if pd.notna(row.get('location')):
            parts.append(f"Location: {row['location']}")
        
        # Work type
        if pd.notna(row.get('formatted_work_type')):
            parts.append(f"Work Type: {row['formatted_work_type']}")
        
        # Experience level
        if pd.notna(row.get('formatted_experience_level')):
            parts.append(f"Experience Level: {row['formatted_experience_level']}")
        
        # Salary
        if pd.notna(row.get('max_salary')) and pd.notna(row.get('min_salary')):
            salary_range = f"${int(row['min_salary']):,} - ${int(row['max_salary']):,}"
            if pd.notna(row.get('pay_period')):
                salary_range += f" {row['pay_period']}"
            parts.append(f"Salary: {salary_range}")
        
        # Required skills
        if pd.notna(row.get('required_skills')) and row['required_skills']:
            parts.append(f"Required Skills: {row['required_skills']}")
        
        # Main description
        if pd.notna(row.get('description')):
            parts.append(f"\nJob Description:\n{row['description']}")
        
        return "\n".join(parts)
    
    def _build_metadata(self, row: pd.Series) -> Dict:
        """Build metadata dictionary"""
        metadata = {
            'job_id': str(row['job_id']),
            'title': str(row.get('title', 'Unknown')),
            'company': str(row.get('company_name') or row.get('name_company', 'Unknown')),
            'location': str(row.get('location', 'Unknown')),
        }
        
        # Optional fields
        if pd.notna(row.get('formatted_work_type')):
            metadata['work_type'] = str(row['formatted_work_type'])
        
        if pd.notna(row.get('formatted_experience_level')):
            metadata['experience_level'] = str(row['formatted_experience_level'])
        
        if pd.notna(row.get('remote_allowed')):
            metadata['remote_allowed'] = bool(row['remote_allowed'])
        
        if pd.notna(row.get('max_salary')):
            metadata['salary_max'] = float(row['max_salary'])
        
        if pd.notna(row.get('min_salary')):
            metadata['salary_min'] = float(row['min_salary'])
        
        return metadata


# Testing
if __name__ == "__main__":
    print("=== Testing Data Processor ===\n")
    
    try:
        processor = DataProcessor()
        
        # Process first 100 jobs for testing
        jobs = processor.process_jobs(max_jobs=100, clean=True)
        
        # Show sample job
        if jobs:
            print("\n📄 Sample Job Document:")
            print("="*60)
            sample = jobs[0]
            print(f"ID: {sample['id']}")
            print(f"\nMetadata: {sample['metadata']}")
            print(f"\nDescription (first 300 chars):\n{sample['description'][:300]}...")
            print("="*60)
        
        print(f"\n✅ Successfully processed {len(jobs)} jobs!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()