#!/usr/bin/env python3
"""
AI Resume Evaluator - CLI Entry Point
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src.evaluator import ResumeEvaluator
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Resume Evaluator - Evaluate candidate-job fit using RAG + LLM"
    )
    
    # Input arguments
    parser.add_argument(
        "--jd",
        type=str,
        required=True,
        help="Path to job description file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        required=True,
        help="Path to resume file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_result.json",
        help="Output file path for results (default: evaluation_result.json)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Text chunk size (default: 500)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # Validate input files
    jd_path = Path(args.jd)
    resume_path = Path(args.resume)
    
    if not jd_path.exists():
        logger.error(f"Job description file not found: {jd_path}")
        sys.exit(1)
    
    if not resume_path.exists():
        logger.error(f"Resume file not found: {resume_path}")
        sys.exit(1)
    
    # Read input files
    logger.info(f"Reading job description from: {jd_path}")
    jd_text = jd_path.read_text(encoding="utf-8")
    
    logger.info(f"Reading resume from: {resume_path}")
    resume_text = resume_path.read_text(encoding="utf-8")
    
    # Create configuration
    config = Config(
        model_name=args.model,
        device=args.device,
        chunk_size=args.chunk_size,
        top_k_retrieval=args.top_k
    )
    
    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = ResumeEvaluator(config=config)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    try:
        result = evaluator.evaluate(
            jd_text=jd_text,
            resume_text=resume_text
        )
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"\nDecision: {result['decision']}")
        print(f"Fit Score: {result['fit_score']}/100")
        print(f"\nStrengths:")
        for strength in result.get('strengths', []):
            print(f"  • {strength}")
        print(f"\nGaps:")
        for gap in result.get('gaps', []):
            print(f"  • {gap}")
        print(f"\nSummary: {result.get('summary', 'N/A')}")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
