"""
Main Resume Evaluator Class
"""
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain.schema import Document
from langgraph.graph import StateGraph, END

from src.config import Config
from src.models.llm import LLMManager
from src.models.embeddings import EmbeddingManager
from src.rag.chunker import TextChunker
from src.rag.vectorstore import VectorStoreManager
from src.agents.requirement_extractor import RequirementExtractor
from src.agents.resume_evaluator import ResumeEvaluatorAgent
from src.agents.decision_maker import DecisionMaker
from src.utils.logger import setup_logger
from src.utils.types import AgentState

logger = setup_logger(__name__)


class ResumeEvaluator:
    """
    Main evaluator class that orchestrates the evaluation workflow.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()
        
        logger.info("Initializing Resume Evaluator...")
        
        # Initialize components
        self.llm_manager = LLMManager(self.config)
        self.embedding_manager = EmbeddingManager(self.config)
        self.chunker = TextChunker(self.config)
        self.vectorstore_manager = VectorStoreManager(self.embedding_manager)
        
        # Initialize agents
        self.requirement_extractor = RequirementExtractor(self.llm_manager)
        self.resume_evaluator_agent = ResumeEvaluatorAgent(self.llm_manager)
        self.decision_maker = DecisionMaker(self.llm_manager)
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        logger.info("✅ Evaluator initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("ingest", self._ingest_node)
        workflow.add_node("index", self._index_node)
        workflow.add_node("extract_requirements", self._extract_requirements_node)
        workflow.add_node("evaluate_resume", self._evaluate_resume_node)
        workflow.add_node("final_decision", self._final_decision_node)
        
        # Define edges
        workflow.set_entry_point("ingest")
        workflow.add_edge("ingest", "index")
        workflow.add_edge("index", "extract_requirements")
        workflow.add_edge("extract_requirements", "evaluate_resume")
        workflow.add_edge("evaluate_resume", "final_decision")
        workflow.add_edge("final_decision", END)
        
        return workflow.compile()
    
    def evaluate(self, jd_text: str, resume_text: str) -> Dict[str, Any]:
        """
        Evaluate a resume against a job description.
        
        Args:
            jd_text: Job description text
            resume_text: Resume text
        
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting evaluation workflow")
        start_time = time.time()
        
        # Initialize state
        initial_state = {
            'jd_text': jd_text,
            'resume_text': resume_text,
            'jd_docs': None,
            'resume_docs': None,
            'jd_vs': None,
            'resume_vs': None,
            'requirements': None,
            'evaluations': None,
            'final_report': None
        }
        
        try:
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            processing_time = time.time() - start_time
            logger.info(f"Evaluation completed in {processing_time:.2f}s")
            
            # Add metadata to report
            report = final_state['final_report']
            report['processing_time_seconds'] = round(processing_time, 2)
            report['timestamp'] = datetime.now().isoformat()
            
            return report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise
    
    def _ingest_node(self, state: AgentState) -> AgentState:
        """Ingest and chunk documents."""
        logger.info("📄 Ingesting documents...")
        
        # Chunk job description
        jd_docs = self.chunker.chunk_text(state['jd_text'])
        logger.info(f"  ✓ JD chunked into {len(jd_docs)} pieces")
        
        # Chunk resume
        resume_docs = self.chunker.chunk_text(state['resume_text'])
        logger.info(f"  ✓ Resume chunked into {len(resume_docs)} pieces")
        
        state['jd_docs'] = jd_docs
        state['resume_docs'] = resume_docs
        
        return state
    
    def _index_node(self, state: AgentState) -> AgentState:
        """Build vector indexes."""
        logger.info("🔍 Building vector indexes...")
        
        # Build JD index
        jd_vs = self.vectorstore_manager.build_index(state['jd_docs'])
        logger.info(f"  ✓ JD index built")
        
        # Build resume index
        resume_vs = self.vectorstore_manager.build_index(state['resume_docs'])
        logger.info(f"  ✓ Resume index built")
        
        state['jd_vs'] = jd_vs
        state['resume_vs'] = resume_vs
        
        return state
    
    def _extract_requirements_node(self, state: AgentState) -> AgentState:
        """Extract requirements from job description."""
        logger.info("🔍 Extracting requirements...")
        
        requirements = self.requirement_extractor.extract(state['jd_text'])
        
        logger.info(f"  ✓ Extracted {len(requirements)} requirements")
        state['requirements'] = requirements
        
        return state
    
    def _evaluate_resume_node(self, state: AgentState) -> AgentState:
        """Evaluate resume against requirements."""
        logger.info("⚖️  Evaluating resume...")
        
        evaluations = self.resume_evaluator_agent.evaluate_all(
            requirements=state['requirements'],
            resume_vs=state['resume_vs']
        )
        
        logger.info(f"  ✓ Completed {len(evaluations)} evaluations")
        state['evaluations'] = evaluations
        
        return state
    
    def _final_decision_node(self, state: AgentState) -> AgentState:
        """Generate final decision."""
        logger.info("📊 Generating final decision...")
        
        final_report = self.decision_maker.make_decision(state['evaluations'])
        
        logger.info(f"  ✓ Decision: {final_report['decision']} ({final_report['fit_score']}/100)")
        state['final_report'] = final_report
        
        return state
