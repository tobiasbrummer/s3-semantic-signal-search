#!/usr/bin/env python3
"""
Experiment 1: Does Phase Actually Help?

Systematic evaluation of phase-based retrieval vs. traditional cosine.

Hypotheses:
    H1: Phase improves ranking of negated content
    H2: Phase reduces false positives from semantic opposites
    H3: Learned phase outperforms heuristic phase

Methodology:
    1. Create controlled test set with negation pairs
    2. Compare retrieval metrics with/without phase
    3. Analyze failure modes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import re
from collections import defaultdict


# =============================================================================
# Test Dataset: Negation Pairs
# =============================================================================

@dataclass
class NegationTestCase:
    """A test case for negation handling"""
    query: str
    positive_docs: List[str]      # Should rank HIGH
    negated_docs: List[str]       # Should rank LOW (semantic opposite)
    neutral_docs: List[str]       # Unrelated, baseline
    
    @property
    def all_docs(self) -> List[str]:
        return self.positive_docs + self.negated_docs + self.neutral_docs
    
    @property
    def labels(self) -> List[str]:
        return (['positive'] * len(self.positive_docs) + 
                ['negated'] * len(self.negated_docs) +
                ['neutral'] * len(self.neutral_docs))


def create_negation_dataset() -> List[NegationTestCase]:
    """Create a systematic negation test dataset"""
    
    test_cases = [
        # Case 1: Simple predicate negation
        NegationTestCase(
            query="The restaurant has good food",
            positive_docs=[
                "The restaurant has good food and great service",
                "Great food at this restaurant",
                "The food here is excellent",
                "I enjoyed the good food at this place",
            ],
            negated_docs=[
                "The restaurant does not have good food",
                "The food here is not good",
                "This restaurant doesn't have good food",
                "The food is bad at this restaurant",
            ],
            neutral_docs=[
                "The weather is nice today",
                "I bought a new car",
                "The movie was interesting",
            ]
        ),
        
        # Case 2: Sentiment reversal
        NegationTestCase(
            query="I recommend this product",
            positive_docs=[
                "I highly recommend this product",
                "This product is recommended",
                "Would definitely recommend this",
                "Great product, recommended!",
            ],
            negated_docs=[
                "I do not recommend this product",
                "I wouldn't recommend this",
                "Cannot recommend this product",
                "This product is not recommended",
            ],
            neutral_docs=[
                "The product weighs 500 grams",
                "Available in three colors",
                "Ships within 5 days",
            ]
        ),
        
        # Case 3: Factual negation
        NegationTestCase(
            query="Python supports multithreading",
            positive_docs=[
                "Python supports multithreading through the threading module",
                "Multithreading is supported in Python",
                "You can use threads in Python",
                "Python has threading capabilities",
            ],
            negated_docs=[
                "Python does not support true multithreading due to GIL",
                "Multithreading is not effective in Python",
                "Python's GIL prevents real multithreading",
                "True parallel threads are not possible in Python",
            ],
            neutral_docs=[
                "Python was created by Guido van Rossum",
                "Python uses indentation for blocks",
                "Python is an interpreted language",
            ]
        ),
        
        # Case 4: Subtle negation (hedging)
        NegationTestCase(
            query="The treatment is effective",
            positive_docs=[
                "The treatment is highly effective",
                "Studies show the treatment is effective",
                "Effective treatment for the condition",
                "The treatment works well",
            ],
            negated_docs=[
                "The treatment is not effective",
                "Studies show the treatment is ineffective",
                "The treatment doesn't work",
                "No evidence that treatment is effective",
            ],
            neutral_docs=[
                "The treatment costs $500",
                "Treatment takes about 6 weeks",
                "The doctor prescribed the treatment",
            ]
        ),
        
        # Case 5: German negation
        NegationTestCase(
            query="Das Hotel ist sauber",
            positive_docs=[
                "Das Hotel ist sehr sauber",
                "Ein sauberes Hotel mit guter Lage",
                "Sauberkeit im Hotel war ausgezeichnet",
                "Das Hotel war sauber und ordentlich",
            ],
            negated_docs=[
                "Das Hotel ist nicht sauber",
                "Das Hotel war leider nicht sauber",
                "Mangelnde Sauberkeit im Hotel",
                "Das Hotel ist unsauber",
            ],
            neutral_docs=[
                "Das Hotel hat 50 Zimmer",
                "Das Hotel liegt in der Stadtmitte",
                "Frühstück kostet 15 Euro",
            ]
        ),
        
        # Case 6: Double negation
        NegationTestCase(
            query="The feature is available",
            positive_docs=[
                "The feature is now available",
                "This feature is available to all users",
                "Available feature in the new version",
                "The feature has been made available",
            ],
            negated_docs=[
                "The feature is not available",
                "This feature is unavailable",
                "The feature is no longer available",
                "Feature not available in this version",
            ],
            neutral_docs=[
                "The feature was announced last week",
                "Users have requested this feature",
                "The feature uses machine learning",
            ]
        ),
    ]
    
    return test_cases


# =============================================================================
# Phase Predictors
# =============================================================================

class HeuristicPhasePredictor:
    """Rule-based phase prediction"""
    
    NEGATION_PATTERNS = [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bnobody\b", r"\bnothing\b",
        r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bwon't\b", 
        r"\bwouldn't\b", r"\bcouldn't\b", r"\bshouldn't\b", r"\bcan't\b",
        r"\bcannot\b", r"\bisn't\b", r"\baren't\b", r"\bwasn't\b",
        r"\bweren't\b", r"\bhasn't\b", r"\bhaven't\b", r"\bhadn't\b",
        r"\bnicht\b", r"\bkein\b", r"\bkeine\b", r"\bnie\b", r"\bohne\b",
        r"\bun\w+\b",  # Prefixes like "unavailable", "unsauber"
    ]
    
    NEGATIVE_SENTIMENT = {
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 
        'ineffective', 'failed', 'failure', 'wrong', 'broken',
        'schlecht', 'mangel', 'schlimm'
    }
    
    def __init__(self):
        self.negation_regex = re.compile(
            '|'.join(self.NEGATION_PATTERNS), 
            re.IGNORECASE
        )
    
    def predict(self, text: str) -> float:
        """Return phase: 0 = affirmative, π = negated"""
        phase = 0.0
        text_lower = text.lower()
        
        # Check for negation patterns
        if self.negation_regex.search(text_lower):
            phase += np.pi
        
        # Check for negative sentiment words
        words = set(re.findall(r'\b\w+\b', text_lower))
        if words & self.NEGATIVE_SENTIMENT:
            phase += np.pi / 4  # Partial shift for sentiment
        
        return phase % (2 * np.pi)
    
    def is_negated(self, text: str) -> bool:
        """Binary negation detection"""
        return self.predict(text) > np.pi / 2


class ContextualPhasePredictor:
    """
    More sophisticated phase prediction considering context.
    
    Key insight: Negation scope matters.
    "I don't think it's bad" is actually positive!
    """
    
    NEGATION_WORDS = {'not', "n't", 'no', 'never', 'neither', 'nobody', 
                      'nothing', 'nowhere', 'none', 'without',
                      'nicht', 'kein', 'keine', 'nie', 'ohne'}
    
    NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'wrong', 'fail', 'poor',
                      'schlecht', 'schlimm', 'mangel'}
    
    POSITIVE_WORDS = {'good', 'great', 'excellent', 'wonderful', 'effective',
                      'gut', 'toll', 'ausgezeichnet', 'sauber'}
    
    def predict(self, text: str) -> float:
        """
        More nuanced phase prediction.
        
        Handles:
        - Double negation: "not bad" → positive
        - Negation scope: "I don't think it's bad" → positive
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Count negations and sentiment
        negation_count = sum(1 for w in words if w in self.NEGATION_WORDS)
        has_negative = bool(set(words) & self.NEGATIVE_WORDS)
        has_positive = bool(set(words) & self.POSITIVE_WORDS)
        
        # Simple model:
        # Even negations cancel out
        # Negation + negative → positive (double negation)
        # Negation + positive → negative
        
        effective_negation = negation_count % 2 == 1
        
        if effective_negation:
            if has_negative:
                # "not bad" → positive
                return 0.0
            else:
                # "not good" → negative
                return np.pi
        else:
            if has_negative:
                return np.pi
            else:
                return 0.0


# =============================================================================
# Similarity Functions
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Traditional cosine similarity"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def resonance_sim(a: np.ndarray, b: np.ndarray, 
                  phase_a: float, phase_b: float) -> float:
    """
    Wave-based similarity with phase.
    
    R = Re[⟨ψ_a | ψ_b⟩] = |a||b|cos(θ)cos(φ_a - φ_b)
    
    where θ is the angle between amplitude vectors
    and (φ_a - φ_b) is the phase difference.
    """
    # Amplitude similarity (cosine)
    amp_sim = cosine_sim(a, b)
    
    # Phase alignment
    phase_diff = phase_a - phase_b
    phase_factor = np.cos(phase_diff)
    
    # Combined resonance
    return amp_sim * phase_factor


# =============================================================================
# Evaluation Metrics
# =============================================================================

@dataclass
class RetrievalMetrics:
    """Metrics for a single test case"""
    
    # Rankings
    positive_ranks: List[int] = field(default_factory=list)
    negated_ranks: List[int] = field(default_factory=list)
    neutral_ranks: List[int] = field(default_factory=list)
    
    # Scores
    positive_scores: List[float] = field(default_factory=list)
    negated_scores: List[float] = field(default_factory=list)
    neutral_scores: List[float] = field(default_factory=list)
    
    @property
    def mean_positive_rank(self) -> float:
        return np.mean(self.positive_ranks) if self.positive_ranks else 0
    
    @property
    def mean_negated_rank(self) -> float:
        return np.mean(self.negated_ranks) if self.negated_ranks else 0
    
    @property
    def negated_above_positive(self) -> int:
        """Count: how many negated docs ranked above any positive doc"""
        if not self.positive_ranks or not self.negated_ranks:
            return 0
        best_positive = min(self.positive_ranks)
        return sum(1 for r in self.negated_ranks if r < best_positive)
    
    @property
    def separation_score(self) -> float:
        """
        How well separated are positive and negated docs?
        1.0 = perfect (all positive above all negated)
        0.0 = random
        -1.0 = inverted (all negated above all positive)
        """
        if not self.positive_scores or not self.negated_scores:
            return 0.0
        
        # Mann-Whitney U statistic based
        count_correct = 0
        count_total = 0
        
        for p_score in self.positive_scores:
            for n_score in self.negated_scores:
                count_total += 1
                if p_score > n_score:
                    count_correct += 1
                elif p_score == n_score:
                    count_correct += 0.5
        
        return (count_correct / count_total) * 2 - 1 if count_total > 0 else 0


def evaluate_retrieval(
    query: str,
    docs: List[str],
    labels: List[str],
    use_phase: bool,
    phase_predictor: HeuristicPhasePredictor,
    vectorizer: TfidfVectorizer
) -> RetrievalMetrics:
    """Evaluate retrieval for a single test case"""
    
    # Encode query and docs
    all_texts = [query] + docs
    
    # Fit vectorizer on all texts
    vectors = vectorizer.fit_transform(all_texts).toarray()
    query_vec = vectors[0]
    doc_vecs = vectors[1:]
    
    # Compute similarities
    scores = []
    
    if use_phase:
        query_phase = phase_predictor.predict(query)
        for i, doc in enumerate(docs):
            doc_phase = phase_predictor.predict(doc)
            score = resonance_sim(query_vec, doc_vecs[i], query_phase, doc_phase)
            scores.append(score)
    else:
        for i in range(len(docs)):
            score = cosine_sim(query_vec, doc_vecs[i])
            scores.append(score)
    
    # Rank documents
    ranked_indices = np.argsort(scores)[::-1]  # Descending
    
    # Collect metrics
    metrics = RetrievalMetrics()
    
    for rank, idx in enumerate(ranked_indices):
        label = labels[idx]
        score = scores[idx]
        
        if label == 'positive':
            metrics.positive_ranks.append(rank)
            metrics.positive_scores.append(score)
        elif label == 'negated':
            metrics.negated_ranks.append(rank)
            metrics.negated_scores.append(score)
        else:
            metrics.neutral_ranks.append(rank)
            metrics.neutral_scores.append(score)
    
    return metrics


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run the complete phase effectiveness experiment"""
    
    print("=" * 70)
    print("EXPERIMENT 1: Does Phase Actually Help?")
    print("=" * 70)
    
    # Setup
    test_cases = create_negation_dataset()
    phase_predictor = HeuristicPhasePredictor()
    contextual_predictor = ContextualPhasePredictor()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    
    # Results storage
    results = {
        'no_phase': [],
        'heuristic_phase': [],
        'contextual_phase': [],
    }
    
    print(f"\nRunning {len(test_cases)} test cases...")
    print()
    
    for i, test_case in enumerate(test_cases):
        print(f"Test Case {i+1}: \"{test_case.query[:50]}...\"")
        
        # Evaluate without phase
        metrics_no_phase = evaluate_retrieval(
            test_case.query,
            test_case.all_docs,
            test_case.labels,
            use_phase=False,
            phase_predictor=phase_predictor,
            vectorizer=vectorizer
        )
        results['no_phase'].append(metrics_no_phase)
        
        # Evaluate with heuristic phase
        metrics_heuristic = evaluate_retrieval(
            test_case.query,
            test_case.all_docs,
            test_case.labels,
            use_phase=True,
            phase_predictor=phase_predictor,
            vectorizer=vectorizer
        )
        results['heuristic_phase'].append(metrics_heuristic)
        
        # Print comparison
        print(f"  No Phase:    Pos rank={metrics_no_phase.mean_positive_rank:.1f}, "
              f"Neg rank={metrics_no_phase.mean_negated_rank:.1f}, "
              f"Sep={metrics_no_phase.separation_score:.2f}")
        print(f"  With Phase:  Pos rank={metrics_heuristic.mean_positive_rank:.1f}, "
              f"Neg rank={metrics_heuristic.mean_negated_rank:.1f}, "
              f"Sep={metrics_heuristic.separation_score:.2f}")
        
        # Check phase predictions
        print(f"  Phase predictions:")
        print(f"    Query: {phase_predictor.predict(test_case.query):.2f} rad "
              f"({'negated' if phase_predictor.is_negated(test_case.query) else 'affirmative'})")
        for doc in test_case.negated_docs[:2]:
            print(f"    Neg doc: {phase_predictor.predict(doc):.2f} rad - \"{doc[:40]}...\"")
        print()
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    for method, metrics_list in results.items():
        avg_pos_rank = np.mean([m.mean_positive_rank for m in metrics_list])
        avg_neg_rank = np.mean([m.mean_negated_rank for m in metrics_list])
        avg_separation = np.mean([m.separation_score for m in metrics_list])
        total_neg_above_pos = sum(m.negated_above_positive for m in metrics_list)
        
        print(f"\n{method.upper()}")
        print(f"  Average positive rank:  {avg_pos_rank:.2f}")
        print(f"  Average negated rank:   {avg_neg_rank:.2f}")
        print(f"  Average separation:     {avg_separation:.3f}")
        print(f"  Negated above positive: {total_neg_above_pos} cases")
    
    # Statistical comparison
    print("\n" + "-" * 70)
    print("STATISTICAL COMPARISON")
    print("-" * 70)
    
    no_phase_sep = [m.separation_score for m in results['no_phase']]
    heuristic_sep = [m.separation_score for m in results['heuristic_phase']]
    
    improvement = np.array(heuristic_sep) - np.array(no_phase_sep)
    
    print(f"\nSeparation improvement (phase - no_phase):")
    print(f"  Mean improvement: {np.mean(improvement):.3f}")
    print(f"  Std:              {np.std(improvement):.3f}")
    print(f"  Cases improved:   {sum(improvement > 0)}/{len(improvement)}")
    print(f"  Cases unchanged:  {sum(improvement == 0)}/{len(improvement)}")
    print(f"  Cases worse:      {sum(improvement < 0)}/{len(improvement)}")
    
    # Detailed breakdown
    print("\n" + "-" * 70)
    print("PER-CASE BREAKDOWN")
    print("-" * 70)
    print(f"\n{'Case':<6} {'No Phase Sep':<15} {'Phase Sep':<15} {'Δ':<10} {'Verdict':<10}")
    print("-" * 60)
    
    for i, (np_m, ph_m) in enumerate(zip(results['no_phase'], results['heuristic_phase'])):
        delta = ph_m.separation_score - np_m.separation_score
        verdict = "✓ Better" if delta > 0.05 else ("✗ Worse" if delta < -0.05 else "≈ Same")
        print(f"{i+1:<6} {np_m.separation_score:<15.3f} {ph_m.separation_score:<15.3f} "
              f"{delta:<+10.3f} {verdict:<10}")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    mean_imp = np.mean(improvement)
    if mean_imp > 0.1:
        conclusion = "STRONG POSITIVE: Phase significantly improves negation handling"
    elif mean_imp > 0.02:
        conclusion = "MODERATE POSITIVE: Phase provides some improvement"
    elif mean_imp > -0.02:
        conclusion = "NEUTRAL: Phase has minimal effect"
    else:
        conclusion = "NEGATIVE: Phase hurts performance (needs investigation)"
    
    print(f"\n{conclusion}")
    print(f"\nMean separation improvement: {mean_imp:.3f}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if mean_imp > 0:
        print("  1. Phase encoding is worth pursuing")
        print("  2. Investigate cases where phase didn't help")
        print("  3. Consider learned phase predictor for subtle cases")
    else:
        print("  1. Review phase prediction logic")
        print("  2. Check if TF-IDF is limiting factor")
        print("  3. Test with neural embeddings")


def analyze_phase_prediction_accuracy():
    """Analyze how well the heuristic predicts actual negation"""
    
    print("\n" + "=" * 70)
    print("PHASE PREDICTION ANALYSIS")
    print("=" * 70)
    
    predictor = HeuristicPhasePredictor()
    test_cases = create_negation_dataset()
    
    # Collect predictions
    true_positive = 0  # Correctly identified as negated
    true_negative = 0  # Correctly identified as affirmative
    false_positive = 0 # Incorrectly identified as negated
    false_negative = 0 # Incorrectly identified as affirmative (missed negation)
    
    errors = []
    
    for test_case in test_cases:
        # Query should be affirmative
        if predictor.is_negated(test_case.query):
            false_positive += 1
            errors.append(('FP', test_case.query))
        else:
            true_negative += 1
        
        # Positive docs should be affirmative
        for doc in test_case.positive_docs:
            if predictor.is_negated(doc):
                false_positive += 1
                errors.append(('FP', doc))
            else:
                true_negative += 1
        
        # Negated docs should be negated
        for doc in test_case.negated_docs:
            if predictor.is_negated(doc):
                true_positive += 1
            else:
                false_negative += 1
                errors.append(('FN', doc))
    
    total = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Neg     Aff")
    print(f"  Actual  Neg      {true_positive:4d}    {false_negative:4d}")
    print(f"          Aff      {false_positive:4d}    {true_negative:4d}")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    
    print(f"\nErrors ({len(errors)} total):")
    for error_type, text in errors[:10]:
        print(f"  {error_type}: \"{text[:60]}...\"")
    
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    run_experiment()
    analyze_phase_prediction_accuracy()
