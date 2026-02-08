"""
Comprehensive Evaluation Metrics for Resume Evaluator

This script calculates multiple metrics to assess system performance:
1. Classification Metrics (Accuracy, Precision, Recall, F1)
2. Requirement-Level Metrics
3. Ranking Metrics (NDCG, MRR)
4. Business Metrics (Cost, Time Savings)
"""
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.evaluator import ResumeEvaluator
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EvaluationMetrics:
    """Calculate comprehensive metrics for the evaluator."""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_classification_metrics(
        self,
        y_true: List[str],
        y_pred: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate standard classification metrics.
        
        Args:
            y_true: Ground truth labels (ACCEPT/MAYBE/REJECT)
            y_pred: Predicted labels
        
        Returns:
            Dictionary with accuracy, precision, recall, F1, etc.
        """
        # Map to binary for some metrics
        binary_true = [1 if y == "ACCEPT" else 0 for y in y_true]
        binary_pred = [1 if y == "ACCEPT" else 0 for y in y_pred]
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "binary_accuracy": accuracy_score(binary_true, binary_pred),
            "binary_precision": precision_score(binary_true, binary_pred, zero_division=0),
            "binary_recall": recall_score(binary_true, binary_pred, zero_division=0),
            "binary_f1": f1_score(binary_true, binary_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["ACCEPT", "MAYBE", "REJECT"]).tolist()
        }
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            labels=["ACCEPT", "MAYBE", "REJECT"],
            output_dict=True
        )
        metrics["classification_report"] = report
        
        return metrics
    
    def calculate_requirement_metrics(
        self,
        evaluations: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate requirement-level metrics.
        
        Measures how well the system evaluates individual requirements.
        
        Args:
            evaluations: Predicted requirement evaluations
            ground_truth: Ground truth requirement evaluations
        
        Returns:
            Dictionary with requirement-level metrics
        """
        if not ground_truth:
            return {
                "requirement_accuracy": 0.0,
                "requirement_precision": 0.0,
                "requirement_recall": 0.0,
                "requirement_f1": 0.0
            }
        
        # Match requirements by text
        matches = []
        for gt in ground_truth:
            for pred in evaluations:
                if self._similar_requirements(gt['requirement'], pred['requirement']):
                    matches.append({
                        'gt_match': gt['match'],
                        'pred_match': pred['match'],
                        'gt_score': gt.get('score', 0),
                        'pred_score': pred.get('score', 0)
                    })
                    break
        
        if not matches:
            return {
                "requirement_accuracy": 0.0,
                "requirement_precision": 0.0,
                "requirement_recall": 0.0,
                "requirement_f1": 0.0
            }
        
        # Calculate metrics
        y_true = [m['gt_match'] for m in matches]
        y_pred = [m['pred_match'] for m in matches]
        
        # Map to binary (yes = 1, no/partial = 0)
        binary_true = [1 if y == "yes" else 0 for y in y_true]
        binary_pred = [1 if y == "yes" else 0 for y in y_pred]
        
        metrics = {
            "requirement_accuracy": accuracy_score(y_true, y_pred),
            "requirement_precision": precision_score(binary_true, binary_pred, zero_division=0),
            "requirement_recall": recall_score(binary_true, binary_pred, zero_division=0),
            "requirement_f1": f1_score(binary_true, binary_pred, zero_division=0),
            "avg_score_mae": np.mean([abs(m['gt_score'] - m['pred_score']) for m in matches])
        }
        
        return metrics
    
    def calculate_ranking_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate ranking metrics (NDCG, MRR, Precision@K).
        
        Useful when ranking multiple candidates.
        
        Args:
            predictions: List of predictions with scores
            ground_truth: List of ground truth labels
        
        Returns:
            Dictionary with ranking metrics
        """
        # Sort by predicted score
        sorted_preds = sorted(predictions, key=lambda x: x['fit_score'], reverse=True)
        
        # Get relevance scores (1 for ACCEPT, 0.5 for MAYBE, 0 for REJECT)
        relevance_map = {"ACCEPT": 1.0, "MAYBE": 0.5, "REJECT": 0.0}
        
        relevance_scores = []
        for pred in sorted_preds:
            # Find matching ground truth
            gt_label = None
            for gt in ground_truth:
                if gt['id'] == pred['id']:
                    gt_label = gt['label']
                    break
            
            if gt_label:
                relevance_scores.append(relevance_map.get(gt_label, 0.0))
        
        if not relevance_scores:
            return {
                "ndcg@5": 0.0,
                "ndcg@10": 0.0,
                "mrr": 0.0,
                "precision@5": 0.0,
                "precision@10": 0.0
            }
        
        # Calculate NDCG@k
        ndcg_5 = self._calculate_ndcg(relevance_scores, k=5)
        ndcg_10 = self._calculate_ndcg(relevance_scores, k=10)
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = self._calculate_mrr(relevance_scores)
        
        # Calculate Precision@k
        precision_5 = self._calculate_precision_at_k(relevance_scores, k=5)
        precision_10 = self._calculate_precision_at_k(relevance_scores, k=10)
        
        return {
            "ndcg@5": ndcg_5,
            "ndcg@10": ndcg_10,
            "mrr": mrr,
            "precision@5": precision_5,
            "precision@10": precision_10
        }
    
    def calculate_business_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        processing_times: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate business metrics (FPR, FNR, cost, time saved).
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            processing_times: Processing times for each evaluation
        
        Returns:
            Dictionary with business metrics
        """
        # Map predictions and ground truth
        y_true = [gt['label'] for gt in ground_truth]
        y_pred = [p['decision'] for p in predictions]
        
        # Binary classification (ACCEPT vs others)
        binary_true = [1 if y == "ACCEPT" else 0 for y in y_true]
        binary_pred = [1 if y == "ACCEPT" else 0 for y in y_pred]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()
        
        # False Positive Rate (wrong accepts - costly!)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate (missed good candidates)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Calculate time metrics
        avg_processing_time = np.mean(processing_times)
        total_processing_time = sum(processing_times)
        
        # Estimate time saved (assume manual review takes 20 minutes)
        manual_time_per_resume = 20 * 60  # 20 minutes in seconds
        time_saved_seconds = len(predictions) * manual_time_per_resume - total_processing_time
        time_saved_hours = time_saved_seconds / 3600
        
        # Cost calculation (API-based systems)
        cost_per_evaluation = 0.00  # Free with local model!
        total_cost = len(predictions) * cost_per_evaluation
        
        # Estimated savings (vs OpenAI GPT-4 at $0.03 per 1K tokens)
        # Assume avg 2K tokens per evaluation
        openai_cost_per_eval = 0.06
        cost_saved = len(predictions) * openai_cost_per_eval
        
        return {
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "avg_processing_time_seconds": round(avg_processing_time, 2),
            "total_processing_time_seconds": round(total_processing_time, 2),
            "time_saved_hours": round(time_saved_hours, 2),
            "cost_per_evaluation_usd": cost_per_evaluation,
            "total_cost_usd": total_cost,
            "cost_saved_vs_openai_usd": round(cost_saved, 2)
        }
    
    def _calculate_ndcg(self, relevance_scores: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        relevance = relevance_scores[:k]
        
        if not relevance:
            return 0.0
        
        # DCG
        dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])
        
        # IDCG (ideal DCG)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance)])
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, relevance_scores: List[float]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, score in enumerate(relevance_scores):
            if score >= 1.0:  # First relevant item
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_precision_at_k(self, relevance_scores: List[float], k: int) -> float:
        """Calculate Precision@K."""
        relevance = relevance_scores[:k]
        if not relevance:
            return 0.0
        return sum([1 for r in relevance if r >= 1.0]) / len(relevance)
    
    def _similar_requirements(self, req1: str, req2: str) -> bool:
        """Check if two requirements are similar (simple string match)."""
        return req1.lower().strip() == req2.lower().strip()


def load_test_dataset(dataset_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load test dataset with ground truth labels.
    
    Expected format:
    {
        "test_cases": [
            {
                "id": "test_1",
                "jd_text": "...",
                "resume_text": "...",
                "ground_truth": {
                    "label": "ACCEPT",
                    "fit_score": 85,
                    "requirements": [...]
                }
            }
        ]
    }
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    test_cases = data['test_cases']
    ground_truth = [tc['ground_truth'] for tc in test_cases]
    
    return test_cases, ground_truth


def run_evaluation(
    dataset_path: str,
    output_path: str,
    config: Config = None
):
    """
    Run comprehensive evaluation on test dataset.
    
    Args:
        dataset_path: Path to test dataset JSON
        output_path: Path to save results
        config: Evaluator configuration
    """
    logger.info("="*80)
    logger.info("STARTING COMPREHENSIVE EVALUATION")
    logger.info("="*80)
    
    # Load dataset
    logger.info(f"Loading test dataset from: {dataset_path}")
    test_cases, ground_truth = load_test_dataset(dataset_path)
    logger.info(f"Loaded {len(test_cases)} test cases")
    
    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = ResumeEvaluator(config=config)
    
    # Run predictions
    logger.info("Running predictions...")
    predictions = []
    processing_times = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Evaluating {i}/{len(test_cases)}: {test_case['id']}")
        
        try:
            result = evaluator.evaluate(
                jd_text=test_case['jd_text'],
                resume_text=test_case['resume_text']
            )
            
            predictions.append({
                'id': test_case['id'],
                'decision': result['decision'],
                'fit_score': result['fit_score'],
                'evaluations': result.get('evaluations', [])
            })
            
            processing_times.append(result.get('processing_time_seconds', 0))
            
        except Exception as e:
            logger.error(f"Failed to evaluate {test_case['id']}: {e}")
            predictions.append({
                'id': test_case['id'],
                'decision': 'ERROR',
                'fit_score': 0,
                'error': str(e)
            })
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics_calculator = EvaluationMetrics()
    
    # 1. Classification metrics
    y_true = [gt['label'] for gt in ground_truth]
    y_pred = [p['decision'] for p in predictions if p['decision'] != 'ERROR']
    
    classification_metrics = metrics_calculator.calculate_classification_metrics(y_true, y_pred)
    
    # 2. Requirement metrics
    requirement_metrics = {
        "avg_requirement_accuracy": 0.0,
        "note": "Requires ground truth requirement evaluations"
    }
    
    # 3. Ranking metrics
    ranking_metrics = metrics_calculator.calculate_ranking_metrics(predictions, ground_truth)
    
    # 4. Business metrics
    business_metrics = metrics_calculator.calculate_business_metrics(
        predictions, ground_truth, processing_times
    )
    
    # Compile all metrics
    all_metrics = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_path,
        "num_test_cases": len(test_cases),
        "classification_metrics": classification_metrics,
        "requirement_metrics": requirement_metrics,
        "ranking_metrics": ranking_metrics,
        "business_metrics": business_metrics,
        "summary": {
            "accuracy": classification_metrics['accuracy'],
            "f1_score": classification_metrics['f1_macro'],
            "avg_processing_time_s": business_metrics['avg_processing_time_seconds'],
            "total_cost_usd": business_metrics['total_cost_usd'],
            "cost_saved_usd": business_metrics['cost_saved_vs_openai_usd']
        }
    }
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nAccuracy: {classification_metrics['accuracy']:.2%}")
    print(f"Precision: {classification_metrics['precision_macro']:.2%}")
    print(f"Recall: {classification_metrics['recall_macro']:.2%}")
    print(f"F1 Score: {classification_metrics['f1_macro']:.2%}")
    print(f"\nFalse Positive Rate: {business_metrics['false_positive_rate']:.2%}")
    print(f"False Negative Rate: {business_metrics['false_negative_rate']:.2%}")
    print(f"\nAvg Processing Time: {business_metrics['avg_processing_time_seconds']:.2f}s")
    print(f"Time Saved: {business_metrics['time_saved_hours']:.1f} hours")
    print(f"Cost Saved: ${business_metrics['cost_saved_vs_openai_usd']:.2f}")
    print("="*80 + "\n")
    
    return all_metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Resume Evaluator Metrics")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to test dataset JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics/evaluation_results.json",
        help="Output path for results"
    )
    
    args = parser.parse_args()
    
    run_evaluation(args.dataset, args.output)


if __name__ == "__main__":
    main()
