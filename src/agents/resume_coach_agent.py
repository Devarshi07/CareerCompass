"""
Resume Coach Agent - Provides evidence-based resume feedback
Analyzes resume structure, content, and optimization
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Any, Optional, List
from src.agents.base_agent import BaseAgent
from src.utils.prompt_templates import PromptTemplates
from src.rag.context_builder import ContextBuilder
from src.rag.retriever import Retriever


class ResumeCoachAgent(BaseAgent):
    """
    Specialized agent for resume feedback and optimization
    """
    
    def __init__(self, llm_provider: str = None, retriever: Retriever = None):
        """
        Initialize resume coach agent with Groq for fast feedback
        
        Args:
            llm_provider: LLM provider (defaults to Groq)
            retriever: Retriever instance (optional, for job-specific feedback)
        """
        super().__init__(llm_provider=llm_provider or 'groq')
        self.retriever = retriever or Retriever()
    
    def get_system_prompt(self) -> str:
        """Get resume coach system prompt"""
        return PromptTemplates.get_resume_coach_prompt()
    
    def process(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Process resume coaching query
        
        Args:
            query: User query
            context: Must contain 'resume_text', optionally 'job_id' or 'job_description'
        
        Returns:
            Resume feedback and recommendations
        """
        # Validate context
        if not self.validate_context(context, ['resume_text']):
            return self.format_error_response(
                "Please provide your resume for review."
            )
        
        resume_text = context['resume_text']
        job_description = context.get('job_description')
        job_id = context.get('job_id')
        focus_areas = context.get('focus_areas')
        
        try:
            # Get job description if job_id provided
            if job_id and not job_description:
                print(f"📄 Fetching job description for ID: {job_id}")
                job = self.retriever.get_job_by_id(job_id)
                if job:
                    job_description = job['document']
                    print("   ✅ Job description loaded")
            
            # Build context for LLM
            context_text = ContextBuilder.build_resume_feedback_context(
                resume_text=resume_text,
                job_description=job_description,
                focus_areas=focus_areas
            )
            
            # Add user query
            full_prompt = f"{context_text}\n\nUser Question: {query}"
            
            # Generate feedback
            print("📝 Analyzing resume...")
            response = self.generate_response(
                user_prompt=full_prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            return response
        
        except Exception as e:
            return self.format_error_response(str(e))
    
    def general_review(self, resume_text: str) -> str:
        """
        Provide general resume review (not job-specific)
        
        Args:
            resume_text: User's resume
        
        Returns:
            General feedback
        """
        context = {'resume_text': resume_text}
        query = "Please review my resume and provide comprehensive feedback on structure, content, and areas for improvement."
        
        return self.process(query, context)
    
    def job_specific_review(
        self,
        resume_text: str,
        job_description: str
    ) -> str:
        """
        Provide job-specific resume optimization
        
        Args:
            resume_text: User's resume
            job_description: Target job description
        
        Returns:
            Job-specific optimization feedback
        """
        context = {
            'resume_text': resume_text,
            'job_description': job_description
        }
        query = "How can I optimize my resume for this specific job? What keywords and skills should I emphasize?"
        
        return self.process(query, context)
    
    def focused_review(
        self,
        resume_text: str,
        focus_areas: List[str]
    ) -> str:
        """
        Provide focused feedback on specific areas
        
        Args:
            resume_text: User's resume
            focus_areas: Specific areas to focus on (e.g., ['experience', 'skills'])
        
        Returns:
            Focused feedback
        """
        context = {
            'resume_text': resume_text,
            'focus_areas': focus_areas
        }
        query = f"Please focus on reviewing: {', '.join(focus_areas)}"
        
        return self.process(query, context)


# Testing
if __name__ == "__main__":
    print("=== Testing Resume Coach Agent ===\n")
    
    try:
        agent = ResumeCoachAgent()
        print(f"✅ Created agent: {agent}\n")
        
        # Test with sample resume
        sample_resume = """
        John Doe
        Software Engineer
        Email: john@example.com
        
        EXPERIENCE:
        - Worked on projects at TechCorp
        - Used Python and JavaScript
        - Helped team with coding
        
        SKILLS:
        Python, JavaScript, HTML, CSS
        
        EDUCATION:
        BS Computer Science
        """
        
        print("Testing general resume review...\n")
        
        context = {'resume_text': sample_resume}
        response = agent.process(
            query="Please review my resume and suggest improvements",
            context=context
        )
        
        print("="*60)
        print(response[:500] + "...")
        print("="*60)
        
        print("\n✅ Resume coach agent test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()