"""
Job Matcher Agent - Finds and ranks matching jobs with explainable scores
Uses semantic search and LLM reasoning for evidence-based recommendations
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import re
from typing import Dict, Any, Optional, List
from src.agents.base_agent import BaseAgent
from src.utils.prompt_templates import PromptTemplates
from src.rag.retriever import Retriever
from src.rag.context_builder import ContextBuilder
from config.settings import settings


class JobMatcherAgent(BaseAgent):
    """
    Specialized agent for job matching and recommendation
    """
    
    def __init__(self, llm_provider: str = None, retriever: Retriever = None):
        """
        Initialize job matcher agent with OpenAI for detailed analysis
        
        Args:
            llm_provider: LLM provider (defaults to OpenAI)
            retriever: Retriever instance for semantic search
        """
        super().__init__(llm_provider=llm_provider or 'openai')
        self.retriever = retriever or Retriever()
    
    def get_system_prompt(self) -> str:
        """Get job matcher system prompt"""
        return PromptTemplates.get_job_matcher_prompt()
    
    def process(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Process job matching query
        
        Args:
            query: User query
            context: Must contain 'resume_text', optionally 'n_results'
        
        Returns:
            Job matching response with recommendations
        """
        # Validate context
        if not self.validate_context(context, ['resume_text']):
            return self.format_error_response(
                "Please provide your resume to find matching jobs."
            )
        
        resume_text = context['resume_text']
        n_results = context.get('n_results', 5)
        min_match_score = context.get('min_match_score', settings.MIN_JOB_MATCH_SCORE)
        
        try:
            # Get total jobs in database
            chroma_manager = self.retriever.chroma_manager
            stats = chroma_manager.get_stats()
            total_jobs = stats.get('total_jobs', 0)
            
            if total_jobs == 0:
                return "I couldn't find any jobs in the database. Please run `python load_data.py` to load job postings."
            
            print(f"🔍 Searching entire database ({total_jobs} jobs) for best matches...")
            
            # Extract key information from resume for better search
            search_query = self._extract_key_resume_info(resume_text)
            print(f"   📋 Search query: {search_query[:200]}...")
            
            # Optimize candidate retrieval - get fewer but more relevant candidates for speed
            # Focus on top semantic matches only - they're most likely to meet threshold
            if total_jobs <= 500:
                max_candidates = min(50, total_jobs)  # Top 50 for small databases (reduced for speed)
                print(f"   📊 Small database detected - retrieving top {max_candidates} most similar jobs...")
            elif total_jobs <= 5000:
                max_candidates = min(50, total_jobs)  # Top 50 for medium databases
                print(f"   📊 Medium database detected - retrieving top {max_candidates} most similar jobs...")
            else:
                max_candidates = min(75, total_jobs)  # Top 75 for large databases
                print(f"   📊 Large database detected - retrieving top {max_candidates} most similar jobs...")
            
            all_job_results = self.retriever.retrieve_jobs(
                query=search_query,
                n_results=max_candidates
            )
            
            if not all_job_results:
                return "I couldn't find any matching jobs in the database. Please try uploading more job postings or adjusting your resume."
            
            # Pre-filter by semantic similarity to speed up LLM analysis
            # Filter out jobs with very low semantic similarity (likely won't meet threshold)
            min_semantic_similarity = 0.35  # 35% semantic similarity minimum (increased for faster filtering)
            pre_filtered_jobs = [
                job for job in all_job_results 
                if job.get('similarity_score', 0) >= min_semantic_similarity
            ]
            
            if len(pre_filtered_jobs) < len(all_job_results):
                print(f"   ⚡ Pre-filtered: {len(pre_filtered_jobs)}/{len(all_job_results)} jobs passed semantic similarity threshold ({min_semantic_similarity*100:.0f}%)")
            
            print(f"   ✅ Retrieved {len(pre_filtered_jobs)} candidate jobs")
            print(f"   🎯 Processing jobs ONE AT A TIME for maximum speed...")
            print(f"   🎯 Will return first {n_results} jobs with match scores ≥ {min_match_score*100:.0f}%")
            
            # Process jobs ONE AT A TIME - fastest approach, stops immediately when found
            matched_jobs = []  # Store (score, job_data, section_text) tuples
            jobs_checked = 0
            
            for job_idx, job in enumerate(pre_filtered_jobs, 1):
                if len(matched_jobs) >= n_results:
                    print(f"   ✅ Found {len(matched_jobs)} jobs above threshold - stopping search")
                    break
                
                jobs_checked += 1
                print(f"   🔄 Checking job {job_idx}/{len(pre_filtered_jobs)}: {job['metadata'].get('title', 'Unknown')} ({len(matched_jobs)}/{n_results} found)...")
                
                # Build context for single job - much smaller context = faster
                context_text = ContextBuilder.build_job_matching_context(
                    resume_text=resume_text,
                    job_results=[job],  # Single job only
                    user_query=query
                )
                
                # Add simplified threshold instruction for speed
                threshold_instruction = f"""

Analyze the COMPLETE resume against this job. Calculate Overall Match Score considering:
- Skills match (30%), Experience level (25%), Education (15%), Projects (15%), Overall fit (15%)

You MUST include: **Overall Match Score:** [X]%

Be thorough - check ALL resume sections against job requirements."""
                context_text += threshold_instruction
                
                try:
                    # Generate response for single job - optimized for speed
                    response = self.generate_response(
                        user_prompt=context_text,
                        temperature=0.7,
                        max_tokens=1000  # Reduced for faster generation
                    )
                    
                    # Parse score from response
                    job_scored = self._parse_job_scores_from_response(response, [job], 1)
                    
                    # Check if this job meets threshold
                    if job_scored:
                        score, job_data, section = job_scored[0]
                        if score >= min_match_score:
                            matched_jobs.append((score, job_data, section))
                            print(f"      ✅ Found match #{len(matched_jobs)}: {score*100:.1f}% - {job_data['metadata'].get('title', 'Unknown')}")
                            
                            # Stop immediately if we have enough
                            if len(matched_jobs) >= n_results:
                                break
                        else:
                            print(f"      ❌ Score: {score*100:.1f}% (below {min_match_score*100:.0f}% threshold)")
                    
                except Exception as e:
                    print(f"      ⚠️  Error processing job {job_idx}: {e}")
                    continue
            
            if not matched_jobs:
                return f"\n⚠️ **No jobs found above {min_match_score*100:.0f}% match threshold.**\n\n" \
                       f"I checked {jobs_checked} jobs from the database (pre-filtered from {len(all_job_results)} candidates), but none met the {min_match_score*100:.0f}% match threshold.\n\n" \
                       "Consider:\n" \
                       "- Expanding your search criteria\n" \
                       "- Adjusting the match threshold\n" \
                       "- Reviewing jobs with lower scores to see if any are still relevant"
            
            # Build final response with jobs in order found (first come, first serve)
            header = f"## 🎯 {len(matched_jobs)} Job Matches Found (First {len(matched_jobs)} Above {min_match_score*100:.0f}% Threshold)\n\n"
            ranked_sections = []
            
            for rank, (score, job_data, section) in enumerate(matched_jobs, 1):
                # Renumber job in section
                section = re.sub(r'###\s*Job\s*#?\d+[:]', f'### Job #{rank}:', section, count=1, flags=re.IGNORECASE)
                ranked_sections.append(section)
                print(f"   📊 Job #{rank}: {score*100:.1f}% match - {job_data['metadata'].get('title', 'Unknown')}")
            
            final_response = header + "".join(ranked_sections)
            print(f"\n   ✅ Returning {len(matched_jobs)} jobs found (checked {jobs_checked} jobs total)")
            
            return final_response
        
        except Exception as e:
            return self.format_error_response(str(e))
    
    def _extract_key_resume_info(self, resume_text: str) -> str:
        """
        Extract key information from resume to create a focused search query
        
        Args:
            resume_text: Full resume text
            
        Returns:
            Focused search query with key skills, experience, and qualifications
        """
        import re
        
        # Extract skills section
        skills_pattern = r'(?:skills?|technical\s+skills?|competencies?|expertise)[:;]?\s*\n?(.+?)(?=\n\n|\n[A-Z][a-z]+:|$)'
        skills_match = re.search(skills_pattern, resume_text, re.IGNORECASE | re.DOTALL)
        skills_text = skills_match.group(1).strip() if skills_match else ""
        
        # Extract experience section
        exp_pattern = r'(?:experience|work\s+experience|employment|professional\s+experience)[:;]?\s*\n?(.+?)(?=\n\n(?:education|skills?|projects|certifications)|$)'
        exp_match = re.search(exp_pattern, resume_text, re.IGNORECASE | re.DOTALL)
        exp_text = exp_match.group(1).strip() if exp_match else ""
        
        # Extract summary/objective
        summary_pattern = r'(?:summary|objective|profile|about)[:;]?\s*\n?(.+?)(?=\n\n(?:experience|education|skills?)|$)'
        summary_match = re.search(summary_pattern, resume_text, re.IGNORECASE | re.DOTALL)
        summary_text = summary_match.group(1).strip() if summary_match else ""
        
        # Build focused query
        query_parts = []
        
        # Add summary if available (usually contains key qualifications)
        if summary_text:
            # Take first 200 chars of summary
            query_parts.append(summary_text[:200])
        
        # Add key skills (first 300 chars)
        if skills_text:
            query_parts.append(skills_text[:300])
        
        # Add experience highlights (first 400 chars)
        if exp_text:
            # Extract job titles and key achievements
            # Look for patterns like "Software Engineer", "Developer", etc.
            job_titles = re.findall(r'\b(?:Senior|Junior|Lead|Principal)?\s*(?:Software|Data|ML|AI|Backend|Frontend|Full\s+Stack)?\s*(?:Engineer|Developer|Scientist|Architect|Analyst|Manager)\b', exp_text, re.IGNORECASE)
            if job_titles:
                query_parts.append(" ".join(set(job_titles[:5])))  # Unique job titles
            
            # Add experience description (first 300 chars)
            query_parts.append(exp_text[:300])
        
        # If we couldn't extract much, use the full resume but limit length
        if not query_parts:
            # Use first 1000 chars of resume as fallback
            query_parts.append(resume_text[:1000])
        
        # Combine and clean up
        search_query = " ".join(query_parts)
        # Remove extra whitespace
        search_query = re.sub(r'\s+', ' ', search_query).strip()
        
        # If query is still too long, truncate to 2000 chars (embedding models handle this well)
        if len(search_query) > 2000:
            search_query = search_query[:2000]
        
        return search_query
    
    def _parse_job_scores_from_response(self, response: str, job_results: List[Dict[str, Any]], start_job_num: int = 1) -> List[tuple]:
        """
        Parse job scores from LLM response and map them to job data
        
        Args:
            response: LLM response with job analyses
            job_results: List of job dictionaries that were analyzed
            start_job_num: Starting job number (for batch processing)
            
        Returns:
            List of tuples: (score, job_data, section_text)
        """
        
        # Score patterns to match different formats
        score_patterns = [
            r'(?:Overall\s+)?Match\s+Score[:\s*]+(\d+(?:\.\d+)?)%',  # Standard format
            r'\*\*Overall\s+Match\s+Score\*\*[:\s]+(\d+(?:\.\d+)?)%',  # Markdown bold
            r'Match\s+Score[:\s]+(\d+(?:\.\d+)?)%',  # Simple format
            r'(\d+(?:\.\d+)?)%\s+match',  # "X% match" format
        ]
        
        # Split response into job sections
        job_sections = re.split(r'###\s+Job\s*#?\d+[:]', response, flags=re.IGNORECASE)
        
        scored_jobs = []
        
        # Process each job section
        for i, section in enumerate(job_sections[1:], 0):  # Skip header
            # Find match score in section
            score_value = None
            for pattern in score_patterns:
                score_match = re.search(pattern, section, re.IGNORECASE)
                if score_match:
                    try:
                        score_value = float(score_match.group(1)) / 100.0
                        break
                    except (ValueError, IndexError):
                        continue
            
            # Map to job data (handle batch offset)
            job_idx = (start_job_num - 1) + i
            if job_idx < len(job_results):
                job_data = job_results[job_idx]
                if score_value is not None:
                    scored_jobs.append((score_value, job_data, section))
                else:
                    # If no score found, assign 0 (will be filtered out)
                    scored_jobs.append((0.0, job_data, section))
        
        return scored_jobs
    
    def _filter_by_match_score(self, response: str, min_score: float, job_results: List[Dict[str, Any]], max_results: int = 5) -> str:
        """
        Filter response to only include jobs above the match score threshold
        
        Args:
            response: LLM-generated response with job matches
            min_score: Minimum match score threshold (0-1)
            job_results: List of job results that were analyzed
            
        Returns:
            Filtered response with only jobs above threshold
        """
        import re
        
        # Extract match scores from response - multiple patterns to catch different formats
        # Pattern 1: "Overall Match Score: X%" or "Match Score: X%"
        # Pattern 2: "**Overall Match Score:** X%" (with markdown bold)
        # Pattern 3: "Match Score: X%" or "Score: X%"
        # Pattern 4: "X% match" or "X% Match"
        score_patterns = [
            r'(?:Overall\s+)?Match\s+Score[:\s*]+(\d+(?:\.\d+)?)%',  # Standard format
            r'\*\*Overall\s+Match\s+Score\*\*[:\s]+(\d+(?:\.\d+)?)%',  # Markdown bold
            r'Match\s+Score[:\s]+(\d+(?:\.\d+)?)%',  # Simple format
            r'(\d+(?:\.\d+)?)%\s+match',  # "X% match" format
        ]
        
        # Find all job sections - handle different formats
        job_sections = re.split(r'###\s+Job\s*#?\d+[:]', response, flags=re.IGNORECASE)
        
        if len(job_sections) <= 1:
            # No job sections found, return as is
            return response
        
        # First section is usually header/intro
        header_section = job_sections[0] if job_sections else ""
        
        # Extract scores from all job sections and filter by threshold
        scored_sections = []
        for i, section in enumerate(job_sections[1:], 1):
            # Try all score patterns to find the match score
            score_value = None
            for pattern in score_patterns:
                score_match = re.search(pattern, section, re.IGNORECASE)
                if score_match:
                    try:
                        score_value = float(score_match.group(1)) / 100.0  # Convert to 0-1 range
                        break
                    except (ValueError, IndexError):
                        continue
            
            if score_value is not None:
                if score_value >= min_score:
                    # Include this job (store with original section and score)
                    scored_sections.append((score_value, section, i))
                    print(f"   ✅ Job #{i}: {score_value*100:.1f}% match (above {min_score*100:.0f}% threshold)")
                else:
                    # Exclude this job
                    print(f"   ❌ Job #{i}: {score_value*100:.1f}% match (below {min_score*100:.0f}% threshold - excluded)")
            else:
                # No score found - exclude for quality
                print(f"   ❌ Job #{i}: No match score found - excluded (LLM may not have followed format)")
        
        # Sort by match score (highest first) - this ensures proper ranking
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        
        # Take top max_results and renumber them based on rank
        top_scored = scored_sections[:max_results]
        top_sections = []
        
        for rank, (score_value, section, original_job_num) in enumerate(top_scored, 1):
            # Remove old job number and add new rank-based number
            # Replace "Job #X:" or "Job X:" with "Job #rank:"
            section = re.sub(r'###\s*Job\s*#?\d+[:]', f'### Job #{rank}:', section, count=1, flags=re.IGNORECASE)
            top_sections.append(section)
            print(f"   📊 Rank #{rank}: {score_value*100:.1f}% match (original Job #{original_job_num})")
        
        # Reconstruct response with header + ranked top sections
        filtered_response = header_section + "".join(top_sections)
        
        # If no jobs passed the threshold, add a message
        if len(top_sections) == 0:
            filtered_response += f"\n\n⚠️ **No jobs found above {min_score*100:.0f}% match threshold.**\n\n"
            filtered_response += f"I analyzed {len(job_results)} jobs from the database, but none met the {min_score*100:.0f}% match threshold.\n\n"
            filtered_response += "Consider:\n"
            filtered_response += "- Expanding your search criteria\n"
            filtered_response += "- Adjusting the match threshold\n"
            filtered_response += "- Reviewing jobs with lower scores to see if any are still relevant"
        else:
            print(f"   ✅ Returning top {len(top_sections)} jobs above {min_score*100:.0f}% threshold")
        
        return filtered_response
    
    def get_top_matches(
        self,
        resume_text: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Get top job matches without LLM analysis (faster)
        
        Args:
            resume_text: User's resume
            n_results: Number of results
        
        Returns:
            Dictionary with job matches
        """
        job_results = self.retriever.retrieve_jobs(
            query=resume_text,
            n_results=n_results
        )
        
        return {
            'matches': job_results,
            'count': len(job_results)
        }
    
    def analyze_specific_job(
        self,
        resume_text: str,
        job_id: str
    ) -> str:
        """
        Analyze match for a specific job
        
        Args:
            resume_text: User's resume
            job_id: Specific job ID to analyze
        
        Returns:
            Detailed match analysis
        """
        # Get specific job
        job = self.retriever.get_job_by_id(job_id)
        
        if not job:
            return f"Job with ID {job_id} not found."
        
        # Create context with single job
        job_result = [{
            'rank': 1,
            'content': job['document'],
            'metadata': job['metadata'],
            'id': job['id'],
            'similarity_score': 1.0
        }]
        
        context_text = ContextBuilder.build_job_matching_context(
            resume_text=resume_text,
            job_results=job_result,
            user_query=f"Analyze my fit for the {job['metadata'].get('title', 'this')} position."
        )
        
        response = self.generate_response(
            user_prompt=context_text,
            temperature=0.7,
            max_tokens=1500
        )
        
        return response


# Testing
if __name__ == "__main__":
    print("=== Testing Job Matcher Agent ===\n")
    
    try:
        from src.vector_store.chroma_manager import ChromaManager
        
        # Initialize
        manager = ChromaManager()
        retriever = Retriever(chroma_manager=manager)
        agent = JobMatcherAgent(retriever=retriever)
        
        print(f"✅ Created agent: {agent}\n")
        
        # Check if we have jobs
        stats = manager.get_stats()
        if stats['total_jobs'] == 0:
            print("⚠️  No jobs in database. Run 'python load_data.py' first.")
        else:
            print(f"📊 Database has {stats['total_jobs']} jobs\n")
            
            # Test with sample resume
            sample_resume = """
            John Doe
            Senior Software Engineer
            
            EXPERIENCE:
            - 5 years of Python development
            - Expert in Django and Flask frameworks
            - AWS cloud infrastructure experience
            - Led team of 5 developers
            - Built scalable REST APIs
            
            SKILLS:
            Python, JavaScript, SQL, AWS, Docker, Kubernetes, CI/CD, Git
            
            EDUCATION:
            BS Computer Science, MIT
            """
            
            context = {
                'resume_text': sample_resume,
                'n_results': 3
            }
            
            print("Testing job matching...\n")
            response = agent.process(
                query="Find the best matching jobs for my background",
                context=context
            )
            
            print("="*60)
            print(response[:500] + "...")
            print("="*60)
            
            print("\n✅ Job matcher agent test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()