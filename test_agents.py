"""
Test script for all agents
Run this to verify all agents are working correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.agents.supervisor_agent import SupervisorAgent
from src.agents.job_matcher_agent import JobMatcherAgent
from src.agents.resume_coach_agent import ResumeCoachAgent
from src.agents.interview_prep_agent import InterviewPrepAgent
from src.vector_store.chroma_manager import ChromaManager
from src.rag.retriever import Retriever


def test_supervisor():
    """Test supervisor agent"""
    print("\n" + "="*70)
    print("🧪 TEST 1: SUPERVISOR AGENT (Routing)")
    print("="*70 + "\n")
    
    try:
        supervisor = SupervisorAgent()
        
        test_queries = [
            "Find jobs that match my resume",
            "Review my resume please",
            "Help me prepare for an interview"
        ]
        
        for query in test_queries:
            agent = supervisor.process(query)
            print(f"✅ '{query[:40]}...' → {agent}")
        
        print("\n✅ Supervisor test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Supervisor test failed: {e}\n")
        return False


def test_job_matcher():
    """Test job matcher agent"""
    print("="*70)
    print("🧪 TEST 2: JOB MATCHER AGENT")
    print("="*70 + "\n")
    
    try:
        # Check if we have jobs
        manager = ChromaManager()
        stats = manager.get_stats()
        
        if stats['total_jobs'] == 0:
            print("⚠️  No jobs in database. Run 'python load_data.py' first.")
            print("   Skipping job matcher test.\n")
            return True
        
        print(f"📊 Database has {stats['total_jobs']} jobs\n")
        
        retriever = Retriever(chroma_manager=manager)
        agent = JobMatcherAgent(retriever=retriever)
        
        sample_resume = """
        John Doe - Senior Python Developer
        
        EXPERIENCE:
        - 5 years Python development
        - Django and Flask expert
        - AWS cloud experience
        
        SKILLS: Python, Django, Flask, AWS, Docker, PostgreSQL
        """
        
        context = {
            'resume_text': sample_resume,
            'n_results': 2
        }
        
        print("Testing job matching...\n")
        response = agent.process("Find matching jobs", context)
        
        print("✅ Response generated:")
        print("-" * 70)
        print(response[:300] + "...")
        print("-" * 70)
        
        print("\n✅ Job matcher test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Job matcher test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_resume_coach():
    """Test resume coach agent"""
    print("="*70)
    print("🧪 TEST 3: RESUME COACH AGENT")
    print("="*70 + "\n")
    
    try:
        agent = ResumeCoachAgent()
        
        sample_resume = """
        Jane Smith
        Software Engineer
        
        EXPERIENCE:
        - Worked on projects
        - Used various technologies
        
        SKILLS:
        Python, Java
        """
        
        context = {'resume_text': sample_resume}
        
        print("Testing resume review...\n")
        response = agent.process("Review my resume", context)
        
        print("✅ Response generated:")
        print("-" * 70)
        print(response[:300] + "...")
        print("-" * 70)
        
        print("\n✅ Resume coach test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Resume coach test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_interview_prep():
    """Test interview prep agent"""
    print("="*70)
    print("🧪 TEST 4: INTERVIEW PREP AGENT")
    print("="*70 + "\n")
    
    try:
        agent = InterviewPrepAgent()
        
        sample_resume = """
        John Doe - Python Developer
        
        EXPERIENCE:
        - 3 years Python development
        - Built REST APIs
        - AWS experience
        """
        
        sample_job = """
        Senior Python Developer
        
        Requirements:
        - 5+ years Python
        - Django experience
        - Cloud infrastructure
        """
        
        context = {
            'resume_text': sample_resume,
            'job_description': sample_job
        }
        
        print("Testing interview prep...\n")
        response = agent.process("Generate interview questions", context)
        
        print("✅ Response generated:")
        print("-" * 70)
        print(response[:300] + "...")
        print("-" * 70)
        
        print("\n✅ Interview prep test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Interview prep test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🚀 TESTING ALL AGENTS")
    print("="*70)
    
    results = {
        'Supervisor': test_supervisor(),
        'Job Matcher': test_job_matcher(),
        'Resume Coach': test_resume_coach(),
        'Interview Prep': test_interview_prep()
    }
    
    # Summary
    print("="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for agent_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{agent_name:20} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\n💡 Next step: Run the Streamlit app")
        print("   streamlit run app/streamlit_app.py\n")
    else:
        print("\n⚠️  Some tests failed. Check errors above.\n")
    
    return all_passed


if __name__ == "__main__":
    main()