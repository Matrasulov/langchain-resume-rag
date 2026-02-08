"""
FastAPI REST API for Resume Evaluator
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import torch
from datetime import datetime

from src.evaluator import ResumeEvaluator
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Resume Evaluator API",
    description="Intelligent resume screening using RAG + Local LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global evaluator instance
evaluator: Optional[ResumeEvaluator] = None


# Request/Response models
class EvaluationRequest(BaseModel):
    """Request model for single evaluation."""
    jd_text: str = Field(..., description="Job description text")
    resume_text: str = Field(..., description="Resume text")
    
    class Config:
        schema_extra = {
            "example": {
                "jd_text": "Senior ML Engineer with 5+ years Python...",
                "resume_text": "John Doe - ML Engineer with 6 years..."
            }
        }


class BatchEvaluationRequest(BaseModel):
    """Request model for batch evaluation."""
    jd_text: str = Field(..., description="Job description text")
    resumes: List[Dict[str, str]] = Field(
        ..., 
        description="List of resumes with id and text"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "jd_text": "Senior ML Engineer...",
                "resumes": [
                    {"id": "candidate_1", "text": "Resume 1..."},
                    {"id": "candidate_2", "text": "Resume 2..."}
                ]
            }
        }


class EvaluationResponse(BaseModel):
    """Response model for evaluation."""
    decision: str
    fit_score: int
    strengths: List[str]
    gaps: List[str]
    summary: str
    total_requirements: int
    critical_met: str
    processing_time_ms: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str]
    version: str
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize the evaluator on startup."""
    global evaluator
    
    logger.info("Initializing Resume Evaluator...")
    
    try:
        config = Config()
        evaluator = ResumeEvaluator(config=config)
        logger.info("✅ Evaluator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Resume Evaluator API")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "AI Resume Evaluator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and model availability.
    """
    return HealthResponse(
        status="healthy" if evaluator is not None else "initializing",
        model_loaded=evaluator is not None,
        gpu_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_candidate(request: EvaluationRequest):
    """
    Evaluate a single candidate against a job description.
    
    **Parameters:**
    - jd_text: Job description text
    - resume_text: Candidate resume text
    
    **Returns:**
    - decision: ACCEPT, MAYBE, or REJECT
    - fit_score: 0-100 score
    - strengths: List of candidate strengths
    - gaps: List of identified gaps
    - summary: Overall assessment
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    start_time = time.time()
    
    try:
        logger.info("Processing evaluation request")
        
        # Run evaluation
        result = evaluator.evaluate(
            jd_text=request.jd_text,
            resume_text=request.resume_text
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"Evaluation completed in {processing_time}ms")
        
        return EvaluationResponse(
            decision=result['decision'],
            fit_score=result['fit_score'],
            strengths=result.get('strengths', []),
            gaps=result.get('gaps', []),
            summary=result.get('summary', ''),
            total_requirements=result.get('total_requirements', 0),
            critical_met=result.get('critical_met', 'N/A'),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", tags=["Evaluation"])
async def batch_evaluate(
    request: BatchEvaluationRequest,
    background_tasks: BackgroundTasks
):
    """
    Evaluate multiple candidates against a job description.
    
    **Parameters:**
    - jd_text: Job description text
    - resumes: List of resume objects with id and text
    
    **Returns:**
    - results: List of evaluation results for each candidate
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    try:
        logger.info(f"Processing batch evaluation for {len(request.resumes)} candidates")
        
        results = []
        
        for resume in request.resumes:
            try:
                result = evaluator.evaluate(
                    jd_text=request.jd_text,
                    resume_text=resume['text']
                )
                
                results.append({
                    "id": resume['id'],
                    "decision": result['decision'],
                    "fit_score": result['fit_score'],
                    "summary": result.get('summary', '')
                })
                
            except Exception as e:
                logger.error(f"Failed to evaluate {resume['id']}: {e}")
                results.append({
                    "id": resume['id'],
                    "error": str(e)
                })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Monitoring"])
async def get_stats():
    """
    Get system statistics.
    
    Returns GPU memory usage and model info.
    """
    stats = {
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        stats.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        })
    
    return stats


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
