"""
LLM Factory - Creates and manages different LLM clients (Groq, OpenAI, Gemini)
Provides a unified interface for interacting with multiple LLM providers.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Optional, Dict, Any

# Try importing each LLM client - make them optional
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq not installed. Install with: pip install groq")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not installed. Install with: pip install openai")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Google Generative AI not installed. Install with: pip install google-generativeai")

from config.settings import settings


class LLMFactory:
    """Factory class to create and manage different LLM clients"""
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM Factory
        
        Args:
            provider: LLM provider ('groq', 'openai', 'gemini'). Defaults to settings.PRIMARY_LLM
            model: Specific model name. Defaults to provider's default model from settings
        """
        self.provider = provider or settings.PRIMARY_LLM
        self.model = model or self._get_default_model()
        self.client = self._initialize_client()
    
    def _get_default_model(self) -> str:
        """Get default model based on provider"""
        model_map = {
            "groq": settings.GROQ_MODEL,
            "openai": settings.OPENAI_MODEL,
            "gemini": settings.GEMINI_MODEL
        }
        return model_map.get(self.provider, settings.GROQ_MODEL)
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Groq package not installed. Run: pip install groq")
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            return Groq(api_key=settings.GROQ_API_KEY)
        
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            openai.api_key = settings.OPENAI_API_KEY
            return openai
        
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai")
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            return genai
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text using the configured LLM
        
        Args:
            prompt: User prompt/query
            system_prompt: System instructions for the model
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Generated text response
        """
        try:
            if self.provider == "groq":
                return self._generate_groq(prompt, system_prompt, temperature, max_tokens, **kwargs)
            elif self.provider == "openai":
                return self._generate_openai(prompt, system_prompt, temperature, max_tokens, **kwargs)
            elif self.provider == "gemini":
                return self._generate_gemini(prompt, system_prompt, temperature, max_tokens, **kwargs)
        except Exception as e:
            raise Exception(f"Error generating response from {self.provider}: {str(e)}")
    
    def _generate_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Groq API"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using OpenAI API"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Google Gemini API"""
        # Remove 'models/' prefix if present
        model_name = self.model.replace('models/', '')
        
        # Combine system prompt and user prompt for Gemini
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        model = genai.GenerativeModel(model_name=model_name)
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Configure safety settings to be less restrictive
        safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }
        
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Check for blocked or empty responses
            if not response.candidates:
                print("⚠️  Gemini: No candidates in response")
                # Try to get more info from prompt_feedback
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    block_reason = response.prompt_feedback.block_reason
                    if block_reason:
                        print(f"⚠️  Gemini: Blocked due to: {block_reason}")
                        return f"I apologize, but the response was blocked by safety filters ({block_reason}). Please try rephrasing your question."
                return "I apologize, but I couldn't generate a complete response. Please try rephrasing your question or providing more details."
            
            # Check if response has parts and text first
            if not response.parts:
                print("⚠️  Gemini: Response has no parts")
                # Check finish reason to provide better error message
                for candidate in response.candidates:
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        if finish_reason == 3:  # SAFETY
                            return "I apologize, but the response was blocked by safety filters. Please try rephrasing your question."
                return "I apologize, but I couldn't generate a complete response. Please try rephrasing your question or providing more details."
            
            # Get the text response
            response_text = response.text
            
            if not response_text or response_text.strip() == "":
                print("⚠️  Gemini: Response text is empty")
                return "I apologize, but I received an empty response. Please try rephrasing your question or providing more details."
            
            # Check finish reason - if MAX_TOKENS, append a note but still return the partial response
            for candidate in response.candidates:
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if finish_reason == 2:  # MAX_TOKENS - response was cut off
                        reason_map = {
                            2: "MAX_TOKENS",
                            3: "SAFETY",
                            4: "RECITATION",
                            5: "OTHER"
                        }
                        reason = reason_map.get(finish_reason, f"UNKNOWN({finish_reason})")
                        print(f"⚠️  Gemini: Finish reason: {reason} - Response was cut off but returning partial content")
                        # Return partial response with helpful note
                        return f"{response_text}\n\n---\n\n*Note: This response was cut off due to length limits. The interview prep guide is very comprehensive! Feel free to ask follow-up questions like 'Tell me more about technical questions' or 'Give me more behavioral questions' to get additional details.*"
                    elif finish_reason == 3:  # SAFETY
                        return "I apologize, but the response was blocked by safety filters. Please try rephrasing your question."
            
            # Normal completion - return full response
            return response_text
            
        except Exception as e:
            error_str = str(e).lower()
            # If generation fails, provide helpful error
            if "quota" in error_str or "rate limit" in error_str:
                raise Exception("Gemini rate limit reached. Please wait a moment and try again.")
            elif "429" in error_str:
                raise Exception("Gemini API rate limit exceeded. Please wait a moment and try again.")
            elif "invalid argument" in error_str or "content" in error_str:
                print(f"⚠️  Gemini error: {e}")
                return "I apologize, but there was an issue with the request. The prompt might be too long or contain invalid content. Please try rephrasing your question or breaking it into smaller parts."
            else:
                print(f"⚠️  Gemini error: {e}")
                raise Exception(f"Gemini API error: {str(e)}")
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate response with retrieved context (RAG pattern)
        
        Args:
            query: User's question/request
            context: Retrieved relevant documents/chunks
            system_prompt: Instructions for the model
            temperature: Sampling temperature
            max_tokens: Maximum response length
        
        Returns:
            Generated response incorporating the context
        """
        # Construct prompt with context
        full_prompt = f"""Context Information:
{context}

User Query: {query}

Please provide a helpful response based on the context above."""
        
        return self.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )


class MultiLLMRouter:
    """
    Router for intelligent multi-LLM selection based on task type.
    Uses different LLMs for different agents to leverage their strengths.
    """
    
    def __init__(self):
        """Initialize multiple LLM clients"""
        self.llms = {}
        
        # Try to initialize available LLMs
        available_providers = []
        
        if GROQ_AVAILABLE and settings.GROQ_API_KEY:
            try:
                self.llms["groq"] = LLMFactory(provider="groq")
                available_providers.append("groq")
            except Exception as e:
                print(f"⚠️  Could not initialize Groq: {e}")
        
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            try:
                self.llms["openai"] = LLMFactory(provider="openai")
                available_providers.append("openai")
            except Exception as e:
                print(f"⚠️  Could not initialize OpenAI: {e}")
        
        if GEMINI_AVAILABLE and settings.GOOGLE_API_KEY:
            try:
                self.llms["gemini"] = LLMFactory(provider="gemini")
                available_providers.append("gemini")
            except Exception as e:
                print(f"⚠️  Could not initialize Gemini: {e}")
        
        if not available_providers:
            raise ValueError("No LLM providers available. Please install packages and configure API keys.")
        
        print(f"✅ Initialized LLMs: {', '.join(available_providers)}")
    
    def route_to_llm(self, agent_type: str) -> LLMFactory:
        """
        Route to appropriate LLM based on agent type
        
        Args:
            agent_type: Type of agent ('supervisor', 'job_matcher', 'resume_coach', 'interview_prep')
        
        Returns:
            Appropriate LLM client for the task
        """
        # Routing strategy (customize based on your needs)
        routing_map = {
            "supervisor": "groq",      # Fast routing decisions
            "job_matcher": "groq",     # Complex reasoning for matching
            "resume_coach": "openai",  # Detailed feedback (if available, else groq)
            "interview_prep": "gemini" # Creative question generation (if available, else groq)
        }
        
        preferred_provider = routing_map.get(agent_type, "groq")
        
        # Fallback to available LLM if preferred is not available
        if preferred_provider in self.llms:
            return self.llms[preferred_provider]
        else:
            # Return first available LLM
            return list(self.llms.values())[0]
    
    def get_all_providers(self) -> list:
        """Get list of all available providers"""
        return list(self.llms.keys())


# Example usage and testing
if __name__ == "__main__":
    # Test LLM Factory
    print("Testing LLM Factory...")
    
    try:
        # Test single LLM
        llm = LLMFactory()
        response = llm.generate(
            prompt="What are the top 3 skills for a software engineer?",
            system_prompt="You are a career advisor assistant.",
            temperature=0.7,
            max_tokens=200
        )
        print(f"\n✅ Single LLM Test ({llm.provider}):")
        print(response)
        
        # Test Multi-LLM Router
        router = MultiLLMRouter()
        print(f"\n✅ Available providers: {router.get_all_providers()}")
        
        # Test routing
        job_matcher_llm = router.route_to_llm("job_matcher")
        print(f"\n✅ Job Matcher routed to: {job_matcher_llm.provider}")
        
    except Exception as e:
        print(f"❌ Error: {e}")