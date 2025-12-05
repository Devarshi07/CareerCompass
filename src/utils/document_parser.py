"""
Document Parser - Extract text from various document formats
Supports PDF, DOCX, and TXT files for resume parsing
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Optional, List, Dict, Any
import re

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  PyPDF2 not installed. Install with: pip install pypdf2")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️  python-docx not installed. Install with: pip install python-docx")


class DocumentParser:
    """
    Parse documents and extract text content.
    Supports PDF, DOCX, and TXT formats.
    """
    
    @staticmethod
    def parse_file(file_path: str) -> str:
        """
        Parse a file and extract text content
        
        Args:
            file_path: Path to the file
        
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and parse accordingly
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return DocumentParser._parse_pdf(file_path)
        elif extension == '.docx':
            return DocumentParser._parse_docx(file_path)
        elif extension == '.txt':
            return DocumentParser._parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    @staticmethod
    def _parse_pdf(file_path: Path) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 is required to parse PDF files")
        
        try:
            reader = PdfReader(str(file_path))
            text = ""
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        
        except Exception as e:
            raise Exception(f"Error parsing PDF: {str(e)}")
    
    @staticmethod
    def _parse_docx(file_path: Path) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required to parse DOCX files")
        
        try:
            doc = Document(str(file_path))
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
        
        except Exception as e:
            raise Exception(f"Error parsing DOCX: {str(e)}")
    
    @staticmethod
    def _parse_txt(file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise Exception(f"Error parsing TXT: {str(e)}")
    
    @staticmethod
    def chunk_text(
        text: str, 
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk (in characters)
            chunk_overlap: Overlap between consecutive chunks
        
        Returns:
            List of text chunks
        """
        if not text or len(text) == 0:
            return []
        
        # Clean the text
        text = DocumentParser.clean_text(text)
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # At least 50% through chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - chunk_overlap
        
        return [c for c in chunks if c]  # Remove empty chunks
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        # Normalize line breaks
        text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
    
    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """
        Extract common resume sections
        
        Args:
            text: Resume text
        
        Returns:
            Dictionary with section names and content
        """
        sections = {}
        
        # Common section headers (case-insensitive)
        section_patterns = {
            'summary': r'(?:professional\s+)?summary|objective|profile',
            'experience': r'(?:work\s+)?experience|employment\s+history|professional\s+experience',
            'education': r'education|academic\s+background|qualifications',
            'skills': r'(?:technical\s+)?skills|competencies|expertise',
            'projects': r'projects|portfolio',
            'certifications': r'certifications?|licenses?',
        }
        
        text_lower = text.lower()
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                start_idx = match.start()
                
                # Find next section or end of text
                next_section_idx = len(text)
                for other_pattern in section_patterns.values():
                    other_match = re.search(other_pattern, text_lower[start_idx + 10:])
                    if other_match:
                        potential_idx = start_idx + 10 + other_match.start()
                        next_section_idx = min(next_section_idx, potential_idx)
                
                # Extract section content
                section_content = text[start_idx:next_section_idx].strip()
                sections[section_name] = section_content
        
        return sections


class ResumeParser:
    """
    Specialized parser for resume documents with metadata extraction
    """
    
    @staticmethod
    def parse_resume(file_path: str) -> Dict[str, Any]:
        """
        Parse a resume and extract structured information
        
        Args:
            file_path: Path to resume file
        
        Returns:
            Dictionary with resume text, sections, and metadata
        """
        # Extract full text
        full_text = DocumentParser.parse_file(file_path)
        
        # Extract sections
        sections = DocumentParser.extract_sections(full_text)
        
        # Chunk the text for embedding
        chunks = DocumentParser.chunk_text(full_text)
        
        # Extract basic metadata
        metadata = ResumeParser._extract_metadata(full_text)
        
        return {
            'full_text': full_text,
            'sections': sections,
            'chunks': chunks,
            'metadata': metadata,
            'file_path': str(file_path)
        }
    
    @staticmethod
    def _extract_metadata(text: str) -> Dict[str, Any]:
        """
        Extract metadata from resume text
        
        Args:
            text: Resume text
        
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            metadata['email'] = email_match.group()
        
        # Extract phone number (US format)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            metadata['phone'] = phone_match.group()
        
        # Extract years of experience (rough estimate)
        exp_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience:\s*(\d+)\+?\s*years?'
        ]
        for pattern in exp_patterns:
            exp_match = re.search(pattern, text.lower())
            if exp_match:
                metadata['years_of_experience'] = int(exp_match.group(1))
                break
        
        return metadata


# Testing
if __name__ == "__main__":
    print("=== Testing Document Parser ===\n")
    
    # Test text chunking
    sample_text = """
    John Doe is a Senior Software Engineer with 5 years of experience in Python, 
    JavaScript, and cloud technologies. He has worked on multiple enterprise applications
    and has strong expertise in microservices architecture and DevOps practices.
    
    Professional Experience:
    Senior Software Engineer at TechCorp (2020-Present)
    - Led development of cloud-native applications
    - Improved system performance by 40%
    - Mentored junior developers
    
    Software Engineer at StartupXYZ (2018-2020)
    - Built RESTful APIs using Django
    - Implemented CI/CD pipelines
    """
    
    print("Testing text chunking...")
    chunks = DocumentParser.chunk_text(sample_text, chunk_size=200, chunk_overlap=50)
    print(f"✅ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(chunk[:100] + "...")
    
    # Test section extraction
    print("\n\n=== Testing Section Extraction ===")
    sections = DocumentParser.extract_sections(sample_text)
    print(f"✅ Extracted {len(sections)} sections:")
    for section_name, content in sections.items():
        print(f"\n{section_name.upper()}:")
        print(content[:100] + "...")
    
    print("\n✅ All tests passed!")