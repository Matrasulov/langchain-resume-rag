"""
Configuration Management
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Main configuration class."""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cuda"
    load_in_4bit: bool = True
    torch_dtype: str = "float16"
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Optional[str] = None  # Uses device if None
    
    # RAG settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    
    # LLM generation settings
    max_new_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    
    # Evaluation settings
    max_requirements: int = 15
    requirement_batch_size: int = 5
    
    # Scoring weights
    must_have_weight: float = 0.6
    preferred_weight: float = 0.3
    responsibility_weight: float = 0.1
    
    # Decision thresholds
    accept_threshold: int = 70
    maybe_threshold: int = 50
    
    # Paths
    log_dir: str = "logs"
    results_dir: str = "results"
    cache_dir: str = ".cache"
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set embedding device to same as model device if not specified
        if self.embedding_device is None:
            self.embedding_device = self.device
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            device=os.getenv("DEVICE", cls.device),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            top_k_retrieval=int(os.getenv("TOP_K", cls.top_k_retrieval)),
            temperature=float(os.getenv("TEMPERATURE", cls.temperature)),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "chunk_size": self.chunk_size,
            "top_k_retrieval": self.top_k_retrieval,
            "temperature": self.temperature,
            "accept_threshold": self.accept_threshold,
            "maybe_threshold": self.maybe_threshold
        }
