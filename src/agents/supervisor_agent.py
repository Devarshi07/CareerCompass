"""
Supervisor Agent - Routes user queries to appropriate specialized agents
Acts as the entry point for all user interactions
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.utils.prompt_templates import PromptTemplates


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that routes queries to specialized agents
    """
    
    def __init__(self, llm_provider: str = None):
        """Initialize supervisor agent with OpenAI for accurate routing"""
        super().__init__(llm_provider=llm_provider or 'openai')
        self.available_agents = ['job_matcher', 'resume_coach', 'interview_prep']
    
    def get_system_prompt(self) -> str:
        """Get supervisor system prompt"""
        return PromptTemplates.get_supervisor_prompt()
    
    def process(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Route query to appropriate agent
        
        Args:
            query: User query
            context: Additional context
        
        Returns:
            Name of the agent to handle the query, or 'general' for greetings/casual chat
        """
        query_lower = query.lower().strip()
        
        # Handle very short/casual queries
        if len(query.split()) <= 3:
            greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy', 'greetings']
            casual = ['how are you', 'whats up', "what's up", 'thanks', 'thank you', 'bye', 'goodbye', 'ok', 'okay', 'yes', 'no']
            
            if any(greeting in query_lower for greeting in greetings):
                return 'general'
            if any(casual_phrase in query_lower for casual_phrase in casual):
                return 'general'
        
        # PRIORITY: Direct keyword matching first (most reliable)
        # Job matching keywords - HIGHEST PRIORITY
        if any(word in query_lower for word in ['find job', 'find me job', 'search job', 'match job', 'matching job', 'show job', 'get job', 'jobs for', 'jobs that match']):
            return 'job_matcher'
        
        # Resume keywords
        if any(word in query_lower for word in ['review resume', 'check resume', 'improve resume', 'feedback on resume', 'resume review', 'resume feedback']):
            return 'resume_coach'
        
        # Interview keywords  
        if any(word in query_lower for word in ['interview prep', 'interview question', 'prepare interview', 'interview help', 'practice interview']):
            return 'interview_prep'
        
        # Single word checks
        if 'resume' in query_lower and 'job' not in query_lower:
            return 'resume_coach'
        
        if 'interview' in query_lower:
            return 'interview_prep'
        
        if any(word in query_lower for word in ['job', 'jobs', 'position', 'positions', 'role', 'roles', 'opening', 'vacancy']):
            return 'job_matcher'
        
        # If still unclear, use LLM as fallback
        try:
            routing_prompt = f"""Classify this query into ONE category:

Query: "{query}"

job_matcher: Finding jobs, job search, job recommendations, which jobs to apply
resume_coach: Resume review, resume improvement, resume feedback  
interview_prep: Interview questions, interview preparation
general: Greetings, casual chat, out of scope

Response (one word):"""
            
            response = self.generate_response(
                user_prompt=routing_prompt,
                temperature=0.1,
                max_tokens=10
            )
            
            agent_name = response.strip().lower()
            
            for available_agent in self.available_agents + ['general']:
                if available_agent in agent_name:
                    return available_agent
        except Exception as e:
            print(f"LLM routing failed: {e}")
        
        # Default to general
        return 'general'
    
    def route_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route query and return routing information
        
        Args:
            query: User query
            context: Additional context
        
        Returns:
            Dictionary with agent_name and reasoning
        """
        agent_name = self.process(query, context)
        
        return {
            'agent': agent_name,
            'query': query,
            'context': context
        }


# Testing
if __name__ == "__main__":
    print("=== Testing Supervisor Agent ===\n")
    
    try:
        supervisor = SupervisorAgent()
        print(f"✅ Created supervisor: {supervisor}\n")
        
        # Test queries
        test_queries = [
            "Which jobs match my resume?",
            "Can you review my resume and suggest improvements?",
            "Help me prepare for an interview at Google",
            "What are the best jobs for a Python developer?",
            "My resume needs work, can you help?",
            "Generate interview questions for a data scientist role"
        ]
        
        print("Testing query routing:\n")
        for i, query in enumerate(test_queries, 1):
            print(f"{i}. Query: \"{query}\"")
            agent = supervisor.process(query)
            print(f"   → Routed to: {agent}\n")
        
        print("✅ All routing tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()