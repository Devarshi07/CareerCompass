"""
Streamlit App - AI-Powered RAG Job Search & Career Assistant
Main application interface with multi-agent chat system
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.job_matcher_agent import JobMatcherAgent
from src.agents.resume_coach_agent import ResumeCoachAgent
from src.agents.interview_prep_agent import InterviewPrepAgent
from src.utils.document_parser import ResumeParser
from src.vector_store.chroma_manager import ChromaManager
from src.rag.retriever import Retriever


# Page configuration
st.set_page_config(
    page_title="AI Job Search Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-box {
        padding: 1rem;
        background-color: #e8f4f8;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stats-box p {
        color: #1a1a1a !important;
        font-size: 0.95rem;
    }
    .stats-box b {
        color: #0d47a1 !important;
        font-weight: 600;
    }
    /* Better spacing and scrolling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 8rem;
    }
    /* Ensure main content area doesn't truncate */
    .main .block-container {
        max-width: 100% !important;
        overflow: visible !important;
    }
    /* Chat container - allow full content display */
    [data-testid="stChatMessageContainer"] {
        overflow: visible !important;
        max-height: none !important;
    }
    /* Chat input stays at bottom with space */
    .stChatInput {
        position: sticky;
        bottom: 0;
        background-color: #0e1117;
        padding: 1rem 0;
        z-index: 999;
    }
    /* Chat message styling for better readability */
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        /* Ensure full content is visible - no truncation */
        overflow: visible !important;
        max-height: none !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    /* User messages - light blue background */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: #e3f2fd !important;
        border-left: 4px solid #2196f3;
    }
    /* Assistant messages - white/pinkish white background */
    [data-testid="stChatMessage"]:not([data-testid*="user"]) {
        background-color: #fff5f8 !important;
        border-left: 4px solid #f06292;
    }
    /* Make chat text more readable */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] code {
        color: #1a1a1a !important;
        line-height: 1.6;
        font-size: 0.95rem;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: pre-wrap !important;
    }
    /* Ensure markdown content is fully displayed */
    [data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
        overflow: visible !important;
        max-height: none !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    /* Bold text in chat */
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] b {
        color: #000000 !important;
        font-weight: 600;
    }
    /* Headers in chat messages */
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        color: #2c3e50 !important;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    /* Ensure all text content wraps properly */
    [data-testid="stChatMessage"] * {
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = None
    
    if 'resume_uploaded' not in st.session_state:
        st.session_state.resume_uploaded = False
    
    if 'agents_initialized' not in st.session_state:
        st.session_state.agents_initialized = False
    
    if 'chroma_manager' not in st.session_state:
        st.session_state.chroma_manager = None
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None


@st.cache_resource(show_spinner=False)
def initialize_agents():
    """Initialize all agents (cached to prevent reloading)"""
    try:
        # Initialize ChromaDB and retriever
        chroma_manager = ChromaManager()
        retriever = Retriever(chroma_manager=chroma_manager)
        
        # Multi-LLM Setup (3 providers):
        # - OpenAI: Supervisor, Job Matcher, Embeddings
        # - Groq: Resume Coach (fast feedback)
        # - Gemini: Interview Prep (creative questions)
        supervisor = SupervisorAgent(llm_provider='openai')
        job_matcher = JobMatcherAgent(llm_provider='openai', retriever=retriever)
        resume_coach = ResumeCoachAgent(llm_provider='groq', retriever=retriever)
        interview_prep = InterviewPrepAgent(llm_provider='gemini', retriever=retriever)
        
        return {
            'supervisor': supervisor,
            'job_matcher': job_matcher,
            'resume_coach': resume_coach,
            'interview_prep': interview_prep,
            'chroma_manager': chroma_manager,
            'retriever': retriever
        }
    except Exception as e:
        st.error(f"Error initializing agents: {e}")
        return None


def parse_resume(uploaded_file):
    """Parse uploaded resume file"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_resume.{uploaded_file.name.split('.')[-1]}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Parse resume
        parsed_resume = ResumeParser.parse_resume(temp_path)
        
        # Clean up temp file
        Path(temp_path).unlink()
        
        return parsed_resume['full_text']
    
    except Exception as e:
        st.error(f"Error parsing resume: {e}")
        return None


def process_query(query, agents, resume_text):
    """Process user query through multi-agent system"""
    try:
        # Route query to appropriate agent
        agent_name = agents['supervisor'].process(query)
        
        # Handle general/conversational queries
        if agent_name == 'general':
            query_lower = query.lower().strip()
            
            # Greetings
            if any(word in query_lower for word in ['hi', 'hello', 'hey', 'morning', 'evening']):
                response = """Hello! 👋 Welcome to your AI-Powered Job Search Assistant!

I'm here to help you with:
🎯 **Job Matching** - Find jobs that match your skills and experience
📝 **Resume Review** - Get feedback and optimization tips  
💼 **Interview Prep** - Practice with personalized questions

What would you like to work on today?"""
                return response, 'general'
            
            # How are you / What's up
            elif any(phrase in query_lower for phrase in ['how are you', 'whats up', "what's up", 'how r u']):
                response = """I'm doing great, thanks for asking! 😊

I'm ready to help you with your job search. What can I assist you with?
- Finding matching jobs
- Reviewing your resume
- Preparing for interviews

Just let me know!"""
                return response, 'general'
            
            # Thanks
            elif 'thank' in query_lower:
                return "You're welcome! Feel free to ask if you need anything else. 😊", 'general'
            
            # Goodbye
            elif any(word in query_lower for word in ['bye', 'goodbye', 'see you']):
                return "Goodbye! Best of luck with your job search! 🚀 Feel free to come back anytime.", 'general'
            
            # Out of scope
            elif any(word in query_lower for word in ['weather', 'news', 'sports', 'movie', 'recipe', 'game']):
                response = """I'm specialized in helping with job searches, resumes, and interviews. 

I can't help with that topic, but I'd be happy to:
- Find jobs matching your skills
- Review and improve your resume
- Prepare you for interviews

What would you like to focus on?"""
                return response, 'general'
            
            # General chat about the assistant
            elif any(word in query_lower for word in ['what can you do', 'help me', 'what do you', 'capabilities', 'features']):
                response = """I'm your AI career assistant! Here's what I can do:

🎯 **Job Matcher**
- Find jobs that semantically match your skills
- Provide explainable match scores with evidence
- Compare multiple opportunities

📝 **Resume Coach**  
- Give specific, actionable feedback
- Optimize for ATS and recruiters
- Suggest improvements with examples

💼 **Interview Prep**
- Generate personalized questions
- Provide answer frameworks (STAR method)
- Company-specific preparation

Just ask me naturally, like "Find jobs for me" or "Review my resume"!"""
                return response, 'general'
            
            # Default for other general queries
            else:
                response = """I'm not sure how to help with that specific question.

I specialize in:
• **Job matching** - Ask me "Find jobs that match my resume"
• **Resume review** - Ask me "Review my resume"
• **Interview prep** - Ask me "Help me prepare for an interview"

What would you like to explore?"""
                return response, 'general'
        
        # Build context for specialized agents
        context = {
            'resume_text': resume_text,
            'n_results': 5
        }
        
        # Get response from appropriate agent
        if agent_name == 'job_matcher':
            response = agents['job_matcher'].process(query, context)
        elif agent_name == 'resume_coach':
            response = agents['resume_coach'].process(query, context)
        elif agent_name == 'interview_prep':
            response = agents['interview_prep'].process(query, context)
        else:
            response = "I'm not sure how to help with that. Please try rephrasing your question."
        
        return response, agent_name
    
    except Exception as e:
        return f"Error processing query: {e}", "error"


def display_chat_message(role, content, agent_name=None):
    """Display a chat message"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <b>👤 You:</b><br><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        agent_emoji = {
            'job_matcher': '🎯',
            'resume_coach': '📝',
            'interview_prep': '💼',
            'error': '⚠️'
        }
        emoji = agent_emoji.get(agent_name, '🤖')
        agent_label = agent_name.replace('_', ' ').title() if agent_name else 'Assistant'
        
        # Format response with proper line breaks
        formatted_content = content.replace('\n', '<br>')
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <b>{emoji} {agent_label}:</b><br><br>
            {formatted_content}
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">💼 AI-Powered Job Search Assistant</div>', 
               unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Find matching jobs, optimize your resume, and prepare for interviews with AI</div>', 
               unsafe_allow_html=True)
    
    # Initialize agents
    if not st.session_state.agents_initialized:
        with st.spinner("🚀 Initializing AI agents..."):
            agents = initialize_agents()
            if agents:
                st.session_state.agents = agents
                st.session_state.chroma_manager = agents['chroma_manager']
                st.session_state.retriever = agents['retriever']
                st.session_state.agents_initialized = True
            else:
                st.error("Failed to initialize agents. Please check your configuration.")
                return
    
    agents = st.session_state.agents
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Upload Your Resume")
        
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF, DOCX, or TXT)",
            type=['pdf', 'docx', 'txt'],
            help="Upload your resume to get personalized job recommendations and feedback"
        )
        
        if uploaded_file:
            with st.spinner("📖 Parsing resume..."):
                resume_text = parse_resume(uploaded_file)
                if resume_text:
                    st.session_state.resume_text = resume_text
                    st.session_state.resume_uploaded = True
                    st.success("✅ Resume uploaded successfully!")
                    
                    with st.expander("📝 Preview Resume"):
                        st.text_area("Resume Content", resume_text, height=200, disabled=True)
        
        st.divider()
        
        # Database stats
        st.header("📊 Database Stats")
        if st.session_state.chroma_manager:
            stats = st.session_state.chroma_manager.get_stats()
            st.markdown(f"""
            <div class="stats-box">
                <p style="margin: 0; color: #1a1a1a;"><b>Jobs Available:</b> {stats['total_jobs']}</p>
                <p style="margin: 0.5rem 0 0 0; color: #1a1a1a;"><b>Resumes Stored:</b> {stats['total_resumes']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if stats['total_jobs'] == 0:
                st.warning("⚠️ No jobs in database. Run `python load_data.py` to load job postings.")
        
        st.divider()
        
        # Help section
        st.header("💡 How to Use")
        st.markdown("""
        **1. Upload Resume** 📄
        - Upload your resume in the sidebar
        
        **2. Ask Questions** 💬
        - "Find jobs that match my resume"
        - "Review my resume and suggest improvements"
        - "Help me prepare for an interview"
        
        **3. Get AI-Powered Insights** 🎯
        - Job matches with explainable scores
        - Evidence-based resume feedback
        - Personalized interview questions
        """)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
    
    # Main chat interface
    st.header("💬 Chat with AI Assistant")
    
    # Check if resume is uploaded
    if not st.session_state.resume_uploaded:
        st.info("👈 Please upload your resume in the sidebar to get started!")
    
    # Chat input - process BEFORE displaying messages
    prompt = st.chat_input(
        "Ask me anything about jobs, resumes, or interviews...",
        disabled=not st.session_state.resume_uploaded,
        key="chat_input"
    )
    
    # Process new input FIRST
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt
        })
        
        # Process and get response immediately
        with st.spinner("🤔 Thinking..."):
            response, agent_name = process_query(
                prompt,
                agents,
                st.session_state.resume_text
            )
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'agent_name': agent_name
        })
    
    # Now display ALL messages (including the new ones)
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message['role'] == 'assistant' and 'agent_name' in message:
                agent_emoji = {
                    'job_matcher': '🎯',
                    'resume_coach': '📝',
                    'interview_prep': '💼',
                    'general': '🤖',
                    'error': '⚠️'
                }
                emoji = agent_emoji.get(message['agent_name'], '🤖')
                agent_label = message['agent_name'].replace('_', ' ').title()
                st.markdown(f"**{emoji} {agent_label}**")
            st.markdown(message['content'])


if __name__ == "__main__":
    main()