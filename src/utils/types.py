"""
Type definitions for the evaluator
"""
from typing import TypedDict, List, Dict, Any, Optional
from langchain.vectorstores import FAISS
from langchain.schema import Document


class AgentState(TypedDict):
    """State object passed through LangGraph workflow."""
    jd_text: str
    resume_text: str
    jd_docs: Optional[List[Document]]
    resume_docs: Optional[List[Document]]
    jd_vs: Optional[FAISS]
    resume_vs: Optional[FAISS]
    requirements: Optional[List[Dict[str, Any]]]
    evaluations: Optional[List[Dict[str, Any]]]
    final_report: Optional[Dict[str, Any]]
