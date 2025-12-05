"""
Interview Prep Agent - Generates personalized interview questions
Creates role-specific questions based on candidate background and target job
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Any, Optional
from src.agents.base_agent import BaseAgent
from src.utils.prompt_templates import PromptTemplates
from src.rag.context_builder import ContextBuilder
from src.rag.retriever import Retriever


class InterviewPrepAgent(BaseAgent):
    """
    Specialized agent for interview preparation
    """
    
    def __init__(self, llm_provider: str = None, retriever: Retriever = None):
        """
        Initialize interview prep agent with Gemini for creative questions
        
        Args:
            llm_provider: LLM provider (defaults to Gemini 1.5 Flash)
            retriever: Retriever instance for fetching job descriptions
        """
        super().__init__(llm_provider=llm_provider or 'gemini')
        self.retriever = retriever or Retriever()
    
    def get_system_prompt(self) -> str:
        """Get interview prep system prompt"""
        return PromptTemplates.get_interview_prep_prompt()
    
    def process(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Process interview preparation query
        
        Args:
            query: User query
            context: Must contain 'resume_text' and optionally 'job_description' or 'job_id'
        
        Returns:
            Personalized interview questions and guidance
        """
        # Validate context
        required_keys = ['resume_text']
        if not self.validate_context(context, required_keys):
            return self.format_error_response(
                "Please provide your resume for interview preparation."
            )
        
        resume_text = context['resume_text']
        job_description = context.get('job_description')
        job_id = context.get('job_id')
        company_info = context.get('company_info')
        
        try:
            # If no job specified, generate general interview prep based ONLY on resume
            if not job_description and not job_id:
                print("📄 No specific job provided, generating general interview questions based on resume only...")
                return self._general_interview_prep(resume_text, query)
            
            # Get job description if job_id provided
            elif job_id and not job_description:
                print(f"📄 Fetching job description for ID: {job_id}")
                job = self.retriever.get_job_by_id(job_id)
                if job:
                    job_description = job['document']
                    # Extract company info if available
                    if not company_info and 'company' in job['metadata']:
                        company_info = f"Company: {job['metadata']['company']}"
                    print("   ✅ Job description loaded")
                else:
                    return self.format_error_response(f"Job ID {job_id} not found.")
            
            # Build context for LLM
            context_text = ContextBuilder.build_interview_prep_context(
                resume_text=resume_text,
                job_description=job_description,
                company_info=company_info
            )
            
            # Truncate context if too long (Gemini has input token limits)
            # Rough estimate: 1 token ≈ 4 characters, leave room for system prompt and user query
            # Gemini 1.5 Flash has ~1M token context, but be conservative
            max_input_chars = 50000  # ~12.5k tokens, leaving room for system prompt
            if len(context_text) > max_input_chars:
                print(f"⚠️  Context too long ({len(context_text)} chars), truncating to {max_input_chars} chars")
                context_text = ContextBuilder.truncate_context(context_text, max_tokens=12500)
            
            # Add user query
            full_prompt = f"{context_text}\n\nUser Request: {query}"
            
            # Log prompt length for debugging
            prompt_length = len(full_prompt)
            print(f"📏 Prompt length: {prompt_length} characters (~{prompt_length // 4} tokens)")
            
            # Generate interview prep
            print("🎯 Generating interview questions...")
            # Use maximum tokens for Gemini 2.5 Flash (8192) to allow comprehensive responses
            response = self.generate_response(
                user_prompt=full_prompt,
                temperature=0.8,  # Higher temperature for creative question generation
                max_tokens=8192  # Maximum for Gemini 2.5 Flash - allows comprehensive interview prep
            )
            
            return response
        
        except Exception as e:
            return self.format_error_response(str(e))
    
    def _general_interview_prep(self, resume_text: str, query: str) -> str:
        """
        Generate general interview questions based on resume only
        
        Args:
            resume_text: User's resume
            query: User's question
        
        Returns:
            General interview preparation guidance
        """
        general_prompt = f"""Generate comprehensive interview preparation guidance based ONLY on the candidate's resume. NO specific job description is provided, so create general interview prep that applies to roles matching their background.

=== CANDIDATE'S RESUME ===
{resume_text}

=== USER REQUEST ===
{query}

IMPORTANT: Since no specific job role is mentioned, provide GENERAL interview preparation that:
- Is based PURELY on their resume background, skills, and experience
- Applies to roles in their field/industry (inferred from their resume)
- Does NOT reference any specific job title or company
- Focuses on their actual experience level and skill set

Provide:

1. **General Interview Strategy** 
   - Based on their experience level (junior/mid/senior)
   - Relevant to their field/industry (inferred from resume)
   - Key strengths from their background to emphasize

2. **Common Questions for Their Profile**
   - Questions typically asked at their experience level
   - Questions relevant to their field/industry
   - Questions about their specific skills and technologies

3. **Technical Questions**
   - Based on the technologies and skills mentioned in their resume
   - Appropriate for their experience level
   - Questions they should be prepared to answer given their background

4. **Behavioral Questions (STAR Method)**
   - Questions they can answer well using experiences from their resume
   - Reference specific projects/roles from their background
   - Help them prepare stories from their actual experience

5. **Questions to Ask Interviewers**
   - Appropriate for their career stage
   - Relevant to their field and experience level

Make ALL questions directly relevant to their actual background, skills, and experiences mentioned in the resume. Do NOT assume any specific job role."""

        # Truncate resume if too long
        max_input_chars = 50000
        if len(general_prompt) > max_input_chars:
            print(f"⚠️  Prompt too long ({len(general_prompt)} chars), truncating resume")
            # Truncate the resume part while keeping the prompt structure
            resume_start = general_prompt.find("=== CANDIDATE'S RESUME ===")
            if resume_start != -1:
                resume_section = general_prompt[resume_start:]
                truncated_resume = ContextBuilder.truncate_context(resume_section, max_tokens=10000)
                general_prompt = general_prompt[:resume_start] + truncated_resume
        
        print(f"📏 Prompt length: {len(general_prompt)} characters (~{len(general_prompt) // 4} tokens)")
        
        # Use maximum tokens for Gemini 2.5 Flash (8192) to allow comprehensive responses
        response = self.generate_response(
            user_prompt=general_prompt,
            temperature=0.8,
            max_tokens=8192  # Maximum for Gemini 2.5 Flash - allows comprehensive responses
        )
        
        return response
    
    def generate_questions(
        self,
        resume_text: str,
        job_description: str,
        question_count: int = 10
    ) -> str:
        """
        Generate interview questions for a specific role
        
        Args:
            resume_text: User's resume
            job_description: Target job description
            question_count: Number of questions to generate
        
        Returns:
            Interview questions with guidance
        """
        context = {
            'resume_text': resume_text,
            'job_description': job_description
        }
        
        query = f"Generate {question_count} tailored interview questions for this role, including technical, behavioral, and company-specific questions. For each question, explain why it might be asked and how I should approach answering based on my background."
        
        return self.process(query, context)
    
    def practice_specific_question(
        self,
        resume_text: str,
        job_description: str,
        question: str
    ) -> str:
        """
        Get guidance on answering a specific interview question
        
        Args:
            resume_text: User's resume
            job_description: Target job description
            question: Specific interview question
        
        Returns:
            Guidance on answering the question
        """
        context = {
            'resume_text': resume_text,
            'job_description': job_description
        }
        
        query = f"How should I answer this interview question based on my background: '{question}'\n\nProvide a structured answer framework and key points from my experience to highlight."
        
        return self.process(query, context)
    
    def get_company_questions(
        self,
        resume_text: str,
        job_description: str,
        company_info: str
    ) -> str:
        """
        Generate company-specific interview questions
        
        Args:
            resume_text: User's resume
            job_description: Target job description
            company_info: Information about the company
        
        Returns:
            Company-specific questions and preparation tips
        """
        context = {
            'resume_text': resume_text,
            'job_description': job_description,
            'company_info': company_info
        }
        
        query = "Generate company-specific interview questions and provide insights on what this company likely values. What questions should I ask the interviewer?"
        
        return self.process(query, context)


# Testing
if __name__ == "__main__":
    print("=== Testing Interview Prep Agent ===\n")
    
    try:
        agent = InterviewPrepAgent()
        print(f"✅ Created agent: {agent}\n")
        
        # Test with sample data
        sample_resume = """
        John Doe
        Senior Software Engineer
        
        EXPERIENCE:
        - 5 years of Python development at TechCorp
        - Led migration to microservices architecture
        - Built scalable REST APIs handling 10M+ requests/day
        - Mentored junior developers
        
        SKILLS:
        Python, Django, AWS, Docker, Kubernetes, PostgreSQL
        """
        
        sample_job = """
        Senior Backend Engineer
        
        We're seeking a Senior Backend Engineer to join our platform team.
        
        Requirements:
        - 5+ years backend development experience
        - Expert in Python and modern frameworks
        - Experience with cloud infrastructure (AWS preferred)
        - Strong understanding of distributed systems
        - Leadership and mentoring experience
        
        You'll work on:
        - Scaling our API infrastructure
        - Improving system reliability
        - Mentoring engineering team
        """
        
        print("Testing interview question generation...\n")
        
        context = {
            'resume_text': sample_resume,
            'job_description': sample_job
        }
        
        response = agent.process(
            query="Generate interview questions for this role and help me prepare",
            context=context
        )
        
        print("="*60)
        print(response[:500] + "...")
        print("="*60)
        
        print("\n✅ Interview prep agent test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()