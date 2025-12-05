"""
Base Agent - Abstract base class for all agents
Defines common interface and shared functionality
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.llm.llm_factory import LLMFactory
from config.settings import settings


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system
    """
    
    def __init__(self, llm_provider: str = None, model: str = None):
        """
        Initialize base agent
        
        Args:
            llm_provider: LLM provider ('groq', 'openai', 'gemini')
            model: Specific model to use
        """
        self.llm_provider = llm_provider or settings.PRIMARY_LLM
        self.llm = LLMFactory(provider=self.llm_provider, model=model)
        self.agent_name = self.__class__.__name__
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent
        Must be implemented by subclasses
        
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    def process(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Process a query and return response
        Must be implemented by subclasses
        
        Args:
            query: User query
            context: Additional context (resume, jobs, etc.)
        
        Returns:
            Agent response
        """
        pass
    
    def generate_response(
        self,
        user_prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate response using LLM
        
        Args:
            user_prompt: User's prompt/query
            system_prompt: System instructions (uses agent's default if None)
            temperature: Sampling temperature
            max_tokens: Maximum response length
        
        Returns:
            Generated response
        """
        system_prompt = system_prompt or self.get_system_prompt()
        temperature = temperature if temperature is not None else settings.AGENT_TEMPERATURE
        max_tokens = max_tokens or settings.MAX_TOKENS
        
        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        
        except Exception as e:
            error_msg = f"Error generating response from {self.agent_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    def validate_context(self, context: Dict[str, Any], required_keys: list) -> bool:
        """
        Validate that context contains required keys
        
        Args:
            context: Context dictionary
            required_keys: List of required keys
        
        Returns:
            True if valid, False otherwise
        """
        if not context:
            return False
        
        missing_keys = [key for key in required_keys if key not in context]
        
        if missing_keys:
            print(f"⚠️  {self.agent_name} missing required context: {missing_keys}")
            return False
        
        return True
    
    def format_error_response(self, error_msg: str) -> str:
        """
        Format error response for user
        
        Args:
            error_msg: Error message
        
        Returns:
            Formatted error response
        """
        return f"I apologize, but I encountered an issue: {error_msg}\n\nPlease try rephrasing your question or providing more context."
    
    def __repr__(self) -> str:
        """String representation of agent"""
        return f"{self.agent_name}(provider={self.llm_provider})"


# Testing
if __name__ == "__main__":
    print("=== Testing Base Agent ===\n")
    
    # Create a simple test agent
    class TestAgent(BaseAgent):
        def get_system_prompt(self) -> str:
            return "You are a helpful test agent."
        
        def process(self, query: str, context: Dict[str, Any] = None) -> str:
            return self.generate_response(query)
    
    try:
        agent = TestAgent()
        print(f"✅ Created agent: {agent}")
        print(f"   Provider: {agent.llm_provider}")
        
        # Test response generation
        response = agent.process("Say hello in one sentence.")
        print(f"\n✅ Test response: {response}")
        
        # Test context validation
        context = {'resume': 'test', 'job': 'test'}
        is_valid = agent.validate_context(context, ['resume', 'job'])
        print(f"\n✅ Context validation: {is_valid}")
        
        print("\n✅ Base agent tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()