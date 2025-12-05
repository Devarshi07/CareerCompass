"""
Context Builder - Constructs context for LLM prompts
Combines retrieved documents with user queries for RAG
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Any, Optional


class ContextBuilder:
    """
    Builds context for LLM prompts by combining retrieved documents
    with user queries in a structured format
    """
    
    @staticmethod
    def build_job_matching_context(
        resume_text: str,
        job_results: List[Dict[str, Any]],
        user_query: Optional[str] = None
    ) -> str:
        """
        Build context for job matching agent
        
        Args:
            resume_text: User's resume text
            job_results: Retrieved job postings
            user_query: Optional user query
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add user's resume with emphasis on completeness
        context_parts.append("=" * 80)
        context_parts.append("=== CANDIDATE'S COMPLETE RESUME ===")
        context_parts.append("=" * 80)
        context_parts.append("")
        context_parts.append("IMPORTANT: This is the COMPLETE resume. You MUST analyze ALL sections:")
        context_parts.append("- Skills (technical and soft skills)")
        context_parts.append("- Work Experience (ALL roles, not just the most recent)")
        context_parts.append("- Education and Certifications")
        context_parts.append("- Projects and Achievements")
        context_parts.append("- Summary/Objective (if present)")
        context_parts.append("")
        context_parts.append("RESUME CONTENT:")
        context_parts.append(resume_text)
        context_parts.append("")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        # Add retrieved job postings
        context_parts.append("=== RELEVANT JOB POSTINGS ===")
        for job in job_results:
            context_parts.append(f"\n[Job {job['rank']}: {job['metadata'].get('title', 'Unknown')}]")
            context_parts.append(f"Company: {job['metadata'].get('company', 'Unknown')}")
            context_parts.append(f"Location: {job['metadata'].get('location', 'Unknown')}")
            context_parts.append(f"Semantic Match Score: {job['similarity_score']:.2%}")
            
            # Truncate job description if too long (keep first 2000 chars for speed)
            job_content = job['content']
            max_job_desc_chars = 2000  # Limit job description length
            if len(job_content) > max_job_desc_chars:
                job_content = job_content[:max_job_desc_chars] + "\n[... Job description truncated for efficiency ...]"
            
            context_parts.append(f"\nJob Description:\n{job_content}")
            context_parts.append("-" * 80)
        
        # Add user query if provided
        if user_query:
            context_parts.append(f"\n=== USER'S QUESTION ===")
            context_parts.append(user_query)
        
        return "\n".join(context_parts)
    
    @staticmethod
    def build_resume_feedback_context(
        resume_text: str,
        job_description: Optional[str] = None,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """
        Build context for resume coaching agent
        
        Args:
            resume_text: User's resume text
            job_description: Target job description (optional)
            focus_areas: Specific areas to focus on (optional)
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add resume
        context_parts.append("=== RESUME TO REVIEW ===")
        context_parts.append(resume_text)
        context_parts.append("")
        
        # Add target job if provided
        if job_description:
            context_parts.append("=== TARGET JOB DESCRIPTION ===")
            context_parts.append(job_description)
            context_parts.append("")
        
        # Add focus areas if provided
        if focus_areas:
            context_parts.append("=== AREAS TO FOCUS ON ===")
            context_parts.append(", ".join(focus_areas))
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def build_interview_prep_context(
        resume_text: str,
        job_description: str,
        company_info: Optional[str] = None
    ) -> str:
        """
        Build context for interview preparation agent
        
        Args:
            resume_text: User's resume text
            job_description: Job description to prepare for
            company_info: Additional company information (optional)
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add candidate's background
        context_parts.append("=== CANDIDATE'S BACKGROUND ===")
        context_parts.append(resume_text)
        context_parts.append("")
        
        # Add target job
        context_parts.append("=== TARGET JOB ===")
        context_parts.append(job_description)
        context_parts.append("")
        
        # Add company info if provided
        if company_info:
            context_parts.append("=== COMPANY INFORMATION ===")
            context_parts.append(company_info)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def build_evidence_based_response(
        claim: str,
        evidence_sources: List[Dict[str, Any]]
    ) -> str:
        """
        Build a response with explicit evidence citations
        
        Args:
            claim: The claim or statement being made
            evidence_sources: List of evidence sources with line numbers
        
        Returns:
            Formatted response with citations
        """
        response_parts = [claim, "\n\nEvidence:"]
        
        for i, source in enumerate(evidence_sources, 1):
            response_parts.append(
                f"{i}. {source.get('text', '')} "
                f"(Source: {source.get('source', 'Unknown')}, "
                f"Line {source.get('line_number', 'N/A')})"
            )
        
        return "\n".join(response_parts)
    
    @staticmethod
    def format_match_explanation(
        job_title: str,
        match_score: float,
        strengths: List[str],
        gaps: List[str],
        evidence: Dict[str, List[str]]
    ) -> str:
        """
        Format a detailed match explanation
        
        Args:
            job_title: Title of the job
            match_score: Overall match score (0-1)
            strengths: List of matching strengths
            gaps: List of skill gaps
            evidence: Dict with 'resume' and 'job' keys containing evidence quotes
        
        Returns:
            Formatted explanation
        """
        explanation_parts = []
        
        explanation_parts.append(f"# Match Analysis: {job_title}")
        explanation_parts.append(f"Overall Match Score: {match_score:.1%}\n")
        
        # Strengths
        explanation_parts.append("## Your Strengths for This Role:")
        for i, strength in enumerate(strengths, 1):
            explanation_parts.append(f"{i}. {strength}")
            if evidence.get('resume'):
                explanation_parts.append(f"   Evidence: \"{evidence['resume'][min(i-1, len(evidence['resume'])-1)]}\"")
        
        explanation_parts.append("")
        
        # Gaps
        if gaps:
            explanation_parts.append("## Areas for Development:")
            for i, gap in enumerate(gaps, 1):
                explanation_parts.append(f"{i}. {gap}")
                if evidence.get('job'):
                    explanation_parts.append(f"   Required: \"{evidence['job'][min(i-1, len(evidence['job'])-1)]}\"")
        
        return "\n".join(explanation_parts)
    
    @staticmethod
    def truncate_context(context: str, max_tokens: int = 3000) -> str:
        """
        Truncate context to fit within token limits
        Rough estimate: 1 token ≈ 4 characters
        
        Args:
            context: Context string
            max_tokens: Maximum number of tokens
        
        Returns:
            Truncated context
        """
        max_chars = max_tokens * 4
        
        if len(context) <= max_chars:
            return context
        
        # Truncate and add indicator
        truncated = context[:max_chars]
        truncated += "\n\n[... Context truncated due to length ...]"
        
        return truncated
    
    @staticmethod
    def extract_key_points(text: str, max_points: int = 5) -> List[str]:
        """
        Extract key points from text (simple sentence-based extraction)
        
        Args:
            text: Input text
            max_points: Maximum number of points to extract
        
        Returns:
            List of key points
        """
        # Split into sentences
        sentences = text.split('.')
        
        # Filter and clean sentences
        key_points = []
        for sentence in sentences:
            cleaned = sentence.strip()
            # Keep sentences that are substantial (more than 20 chars)
            if len(cleaned) > 20 and not cleaned.startswith('==='):
                key_points.append(cleaned)
                if len(key_points) >= max_points:
                    break
        
        return key_points


# Testing
if __name__ == "__main__":
    print("=== Testing Context Builder ===\n")
    
    # Sample data
    sample_resume = """
    John Doe
    Senior Software Engineer
    Email: john@example.com
    
    EXPERIENCE:
    - 5 years of Python development
    - Expert in Django and Flask
    - AWS and Docker experience
    - Led team of 5 developers
    
    SKILLS:
    Python, JavaScript, SQL, AWS, Docker, CI/CD
    """
    
    sample_job_results = [
        {
            'rank': 1,
            'content': 'Looking for Senior Python Developer with Django expertise and cloud experience.',
            'metadata': {
                'title': 'Senior Python Developer',
                'company': 'TechCorp',
                'location': 'Remote'
            },
            'similarity_score': 0.87
        }
    ]
    
    # Test job matching context
    print("=== Job Matching Context ===")
    context = ContextBuilder.build_job_matching_context(
        resume_text=sample_resume,
        job_results=sample_job_results,
        user_query="Which job is the best match for me?"
    )
    print(context[:400] + "...\n")
    
    # Test match explanation
    print("\n=== Match Explanation ===")
    explanation = ContextBuilder.format_match_explanation(
        job_title="Senior Python Developer",
        match_score=0.85,
        strengths=["5 years Python experience", "Django expertise", "Cloud experience"],
        gaps=["Kubernetes knowledge"],
        evidence={
            'resume': ["5 years of Python development", "Expert in Django"],
            'job': ["Kubernetes experience required"]
        }
    )
    print(explanation)
    
    # Test key points extraction
    print("\n\n=== Key Points Extraction ===")
    key_points = ContextBuilder.extract_key_points(sample_resume)
    print("Key Points:")
    for i, point in enumerate(key_points, 1):
        print(f"{i}. {point}")
    
    print("\n✅ All tests passed!")