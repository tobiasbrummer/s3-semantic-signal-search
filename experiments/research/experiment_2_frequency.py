#!/usr/bin/env python3
"""
Experiment 2: Does Frequency/Abstraction Level Help?

Hypotheses:
    H1: Abstract queries should match abstract documents better
    H2: Specific queries should match specific documents better
    H3: Frequency filtering improves retrieval precision

Methodology:
    1. Create test set with documents at different abstraction levels
    2. Create queries with clear abstraction intent
    3. Compare retrieval with/without frequency filtering
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from enum import Enum


# =============================================================================
# Abstraction Level Definition
# =============================================================================

class AbstractionLevel(Enum):
    """Semantic abstraction levels"""
    VERY_ABSTRACT = 1   # Conceptual, definitional
    ABSTRACT = 2        # General explanations
    MIXED = 3           # Both abstract and concrete
    SPECIFIC = 4        # Detailed, examples
    VERY_SPECIFIC = 5   # Code, numbers, exact data


@dataclass
class AbstractionTestCase:
    """Test case for abstraction-level retrieval"""
    query: str
    query_level: AbstractionLevel
    documents: List[Tuple[str, AbstractionLevel]]  # (text, level) pairs
    
    @property
    def doc_texts(self) -> List[str]:
        return [d[0] for d in self.documents]
    
    @property
    def doc_levels(self) -> List[AbstractionLevel]:
        return [d[1] for d in self.documents]


# =============================================================================
# Test Dataset
# =============================================================================

def create_abstraction_dataset() -> List[AbstractionTestCase]:
    """Create test cases spanning abstraction levels"""
    
    test_cases = [
        # Case 1: Machine Learning
        AbstractionTestCase(
            query="What is machine learning?",
            query_level=AbstractionLevel.VERY_ABSTRACT,
            documents=[
                ("Machine learning is a subset of artificial intelligence that enables "
                 "systems to learn and improve from experience without being explicitly "
                 "programmed. It focuses on developing algorithms that can access data "
                 "and use it to learn for themselves.",
                 AbstractionLevel.VERY_ABSTRACT),
                
                ("Machine learning algorithms are typically categorized into supervised, "
                 "unsupervised, and reinforcement learning. Supervised learning uses labeled "
                 "data, while unsupervised learning finds patterns in unlabeled data.",
                 AbstractionLevel.ABSTRACT),
                
                ("To implement a basic machine learning model, you first split your data "
                 "into training and test sets, typically 80/20. Then you choose an algorithm "
                 "like random forest or neural network and train it on the training data.",
                 AbstractionLevel.MIXED),
                
                ("from sklearn.model_selection import train_test_split\n"
                 "from sklearn.ensemble import RandomForestClassifier\n"
                 "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
                 "model = RandomForestClassifier(n_estimators=100)\n"
                 "model.fit(X_train, y_train)",
                 AbstractionLevel.VERY_SPECIFIC),
                
                ("In our experiment, we trained a RandomForest with 100 trees on the "
                 "MNIST dataset (60,000 training samples). After 50 epochs with learning "
                 "rate 0.001, we achieved 98.2% accuracy on the test set.",
                 AbstractionLevel.VERY_SPECIFIC),
            ]
        ),
        
        # Case 2: Specific code query
        AbstractionTestCase(
            query="RandomForestClassifier n_estimators parameter sklearn",
            query_level=AbstractionLevel.VERY_SPECIFIC,
            documents=[
                ("Machine learning is a subset of artificial intelligence that enables "
                 "systems to learn and improve from experience without being explicitly "
                 "programmed.",
                 AbstractionLevel.VERY_ABSTRACT),
                
                ("Random forests are an ensemble method that combines multiple decision "
                 "trees. They reduce overfitting by averaging predictions across trees.",
                 AbstractionLevel.ABSTRACT),
                
                ("The n_estimators parameter controls how many trees are in the forest. "
                 "More trees generally improve performance but increase computation time. "
                 "Common values range from 100 to 1000.",
                 AbstractionLevel.SPECIFIC),
                
                ("clf = RandomForestClassifier(n_estimators=500, max_depth=10, "
                 "min_samples_split=5, random_state=42)",
                 AbstractionLevel.VERY_SPECIFIC),
            ]
        ),
        
        # Case 3: Database concepts
        AbstractionTestCase(
            query="What is database normalization?",
            query_level=AbstractionLevel.ABSTRACT,
            documents=[
                ("Databases are organized collections of structured data that enable "
                 "efficient storage, retrieval, and management of information.",
                 AbstractionLevel.VERY_ABSTRACT),
                
                ("Database normalization is a process of organizing data to reduce "
                 "redundancy and improve data integrity. It involves dividing tables "
                 "and establishing relationships between them.",
                 AbstractionLevel.ABSTRACT),
                
                ("First Normal Form (1NF) requires atomic values and no repeating groups. "
                 "Second Normal Form (2NF) requires 1NF plus no partial dependencies. "
                 "Third Normal Form (3NF) requires 2NF plus no transitive dependencies.",
                 AbstractionLevel.MIXED),
                
                ("To normalize the Orders table to 3NF:\n"
                 "1. Split customer info: CREATE TABLE customers (id INT, name VARCHAR);\n"
                 "2. Split product info: CREATE TABLE products (id INT, name VARCHAR);\n"
                 "3. Orders references both: orders(id, customer_id, product_id, quantity);",
                 AbstractionLevel.VERY_SPECIFIC),
            ]
        ),
        
        # Case 4: Cooking (non-technical domain)
        AbstractionTestCase(
            query="How does fermentation work?",
            query_level=AbstractionLevel.ABSTRACT,
            documents=[
                ("Fermentation is a metabolic process that has been used by humans for "
                 "thousands of years to preserve food and create beverages. It represents "
                 "one of the oldest forms of biotechnology.",
                 AbstractionLevel.VERY_ABSTRACT),
                
                ("In fermentation, microorganisms like yeast or bacteria convert sugars "
                 "into alcohol, acids, or gases. This anaerobic process produces energy "
                 "for the organisms while creating useful byproducts for humans.",
                 AbstractionLevel.ABSTRACT),
                
                ("To make sourdough bread, combine flour and water, let wild yeast "
                 "colonize over 5-7 days with daily feedings. The lactobacilli produce "
                 "lactic acid giving the bread its characteristic tang.",
                 AbstractionLevel.MIXED),
                
                ("Recipe: 500g flour, 350g water, 100g starter, 10g salt. "
                 "Mix, autolyse 30min, fold every 30min for 2hrs, "
                 "bulk ferment 4-6hrs at 75°F, shape, proof 12hrs at 38°F, "
                 "bake at 450°F with steam for 20min then 425°F for 25min.",
                 AbstractionLevel.VERY_SPECIFIC),
            ]
        ),
        
        # Case 5: Mixed query
        AbstractionTestCase(
            query="machine learning training process steps",
            query_level=AbstractionLevel.MIXED,
            documents=[
                ("Machine learning represents a paradigm shift in how we approach "
                 "problem-solving with computers.",
                 AbstractionLevel.VERY_ABSTRACT),
                
                ("The machine learning workflow involves data collection, preprocessing, "
                 "model selection, training, evaluation, and deployment. Each step is "
                 "critical for building effective models.",
                 AbstractionLevel.ABSTRACT),
                
                ("Training a model involves: 1) Loading data, 2) Splitting into train/test, "
                 "3) Feature engineering, 4) Model fitting, 5) Hyperparameter tuning, "
                 "6) Evaluation with metrics like accuracy and F1.",
                 AbstractionLevel.MIXED),
                
                ("Epoch 1/100 - loss: 2.3451 - accuracy: 0.1234\n"
                 "Epoch 50/100 - loss: 0.4521 - accuracy: 0.8567\n"
                 "Epoch 100/100 - loss: 0.0891 - accuracy: 0.9723",
                 AbstractionLevel.VERY_SPECIFIC),
            ]
        ),
    ]
    
    return test_cases


# =============================================================================
# Frequency Estimator
# =============================================================================

class FrequencyEstimator:
    """
    Estimate semantic frequency/abstraction level from text.
    
    High frequency (specific) indicators:
    - Numbers, dates, measurements
    - Code patterns
    - Named entities
    - Technical parameters
    
    Low frequency (abstract) indicators:
    - Abstract nouns
    - Hedging language
    - Meta-discussion
    - Generalizations
    """
    
    # Patterns indicating HIGH frequency (specific/concrete)
    HIGH_FREQ_PATTERNS = [
        r'\d+\.?\d*',                    # Numbers
        r'\d{4}[-/]\d{2}[-/]\d{2}',      # Dates
        r'\d+%',                          # Percentages
        r'[A-Z][a-z]+[A-Z]\w*',          # CamelCase (code)
        r'_\w+',                          # Snake_case (code)
        r'\b\d+[a-zA-Z]+\b',             # Measurements (100ml, 5kg)
        r'def |class |import |from ',    # Python code
        r'SELECT|INSERT|CREATE|FROM',    # SQL
        r'[:,;]\s*\n',                   # Code-like formatting
    ]
    
    # Words indicating LOW frequency (abstract/conceptual)
    ABSTRACT_WORDS = {
        'concept', 'theory', 'approach', 'paradigm', 'principle',
        'framework', 'methodology', 'philosophy', 'perspective',
        'generally', 'typically', 'essentially', 'fundamentally',
        'represents', 'enables', 'involves', 'encompasses',
        'definition', 'meaning', 'understanding', 'nature',
    }
    
    # Words indicating HIGH frequency (specific/detailed)
    SPECIFIC_WORDS = {
        'specifically', 'exactly', 'precisely', 'particular',
        'step', 'parameter', 'value', 'setting', 'configuration',
        'example', 'instance', 'case', 'sample',
        'result', 'output', 'accuracy', 'performance',
    }
    
    def __init__(self):
        self.high_freq_regex = re.compile('|'.join(self.HIGH_FREQ_PATTERNS))
    
    def estimate(self, text: str) -> float:
        """
        Estimate frequency from text.
        
        Returns: 0.0 (very abstract) to 1.0 (very specific)
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        word_count = len(text_lower.split())
        
        # Count high-frequency indicators
        high_freq_matches = len(self.high_freq_regex.findall(text))
        specific_word_count = len(words & self.SPECIFIC_WORDS)
        
        # Count low-frequency indicators
        abstract_word_count = len(words & self.ABSTRACT_WORDS)
        
        # Compute score
        high_score = (high_freq_matches * 0.1 + specific_word_count * 0.15)
        low_score = abstract_word_count * 0.2
        
        # Normalize by text length
        if word_count > 0:
            high_score = min(high_score, 1.0)
            low_score = min(low_score, 1.0)
        
        # Combined frequency: high indicators push up, low indicators push down
        frequency = 0.5 + high_score - low_score
        
        return np.clip(frequency, 0.0, 1.0)
    
    def estimate_level(self, text: str) -> AbstractionLevel:
        """Convert frequency to abstraction level"""
        freq = self.estimate(text)
        
        if freq < 0.25:
            return AbstractionLevel.VERY_ABSTRACT
        elif freq < 0.4:
            return AbstractionLevel.ABSTRACT
        elif freq < 0.6:
            return AbstractionLevel.MIXED
        elif freq < 0.8:
            return AbstractionLevel.SPECIFIC
        else:
            return AbstractionLevel.VERY_SPECIFIC


# =============================================================================
# Frequency-Filtered Retrieval
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def frequency_weighted_similarity(
    query_vec: np.ndarray, 
    doc_vec: np.ndarray,
    query_freq: float,
    doc_freq: float,
    alpha: float = 0.3  # How much to weight frequency matching
) -> float:
    """
    Similarity that considers frequency alignment.
    
    score = (1 - α) * cosine + α * frequency_match
    
    where frequency_match = 1 - |query_freq - doc_freq|
    """
    cosine = cosine_similarity(query_vec, doc_vec)
    freq_match = 1 - abs(query_freq - doc_freq)
    
    return (1 - alpha) * cosine + alpha * freq_match


def frequency_filtered_similarity(
    query_vec: np.ndarray,
    doc_vec: np.ndarray,
    query_freq: float,
    doc_freq: float,
    bandwidth: float = 0.3  # How wide the frequency filter is
) -> float:
    """
    Similarity with hard frequency filtering.
    
    Documents outside the frequency band get penalized.
    """
    cosine = cosine_similarity(query_vec, doc_vec)
    
    freq_diff = abs(query_freq - doc_freq)
    
    if freq_diff <= bandwidth:
        # Within band: full score
        return cosine
    else:
        # Outside band: attenuated
        attenuation = np.exp(-(freq_diff - bandwidth) ** 2 / (2 * 0.1 ** 2))
        return cosine * attenuation


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class FrequencyMetrics:
    """Metrics for frequency-based retrieval"""
    
    # Did we retrieve the right abstraction level?
    level_match_at_1: bool = False
    level_match_at_3: bool = False
    
    # How close was the top result's frequency?
    top_freq_diff: float = 0.0
    
    # Average frequency diff in top-3
    avg_top3_freq_diff: float = 0.0
    
    # Ranking of best-level-match document
    best_level_rank: int = 0


def evaluate_frequency_retrieval(
    test_case: AbstractionTestCase,
    use_frequency: bool,
    frequency_mode: str = 'weighted',  # 'weighted' or 'filtered'
    alpha: float = 0.3,
    vectorizer: TfidfVectorizer = None
) -> Tuple[FrequencyMetrics, List[Tuple[str, float, float]]]:
    """Evaluate retrieval with/without frequency"""
    
    freq_estimator = FrequencyEstimator()
    
    # Fit vectorizer
    all_texts = [test_case.query] + test_case.doc_texts
    vectors = vectorizer.fit_transform(all_texts).toarray()
    query_vec = vectors[0]
    doc_vecs = vectors[1:]
    
    # Estimate frequencies
    query_freq = freq_estimator.estimate(test_case.query)
    doc_freqs = [freq_estimator.estimate(d) for d in test_case.doc_texts]
    
    # Compute similarities
    results = []
    for i, doc in enumerate(test_case.doc_texts):
        if use_frequency:
            if frequency_mode == 'weighted':
                score = frequency_weighted_similarity(
                    query_vec, doc_vecs[i], query_freq, doc_freqs[i], alpha
                )
            else:
                score = frequency_filtered_similarity(
                    query_vec, doc_vecs[i], query_freq, doc_freqs[i]
                )
        else:
            score = cosine_similarity(query_vec, doc_vecs[i])
        
        results.append((doc, score, doc_freqs[i], test_case.doc_levels[i]))
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Compute metrics
    metrics = FrequencyMetrics()
    
    # Query level (target)
    target_level = test_case.query_level
    
    # Level match at 1 and 3
    metrics.level_match_at_1 = results[0][3] == target_level
    metrics.level_match_at_3 = any(r[3] == target_level for r in results[:3])
    
    # Frequency differences
    metrics.top_freq_diff = abs(query_freq - results[0][2])
    metrics.avg_top3_freq_diff = np.mean([abs(query_freq - r[2]) for r in results[:3]])
    
    # Rank of best level match
    for rank, r in enumerate(results):
        if r[3] == target_level:
            metrics.best_level_rank = rank
            break
    else:
        metrics.best_level_rank = len(results)
    
    return metrics, [(r[0][:50], r[1], r[2]) for r in results]


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run the frequency/abstraction experiment"""
    
    print("=" * 70)
    print("EXPERIMENT 2: Does Frequency/Abstraction Level Help?")
    print("=" * 70)
    
    test_cases = create_abstraction_dataset()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    freq_estimator = FrequencyEstimator()
    
    print(f"\nRunning {len(test_cases)} test cases...")
    print()
    
    # Results storage
    results = {
        'no_freq': [],
        'weighted': [],
        'filtered': [],
    }
    
    for i, test_case in enumerate(test_cases):
        query_freq = freq_estimator.estimate(test_case.query)
        query_level = freq_estimator.estimate_level(test_case.query)
        
        print(f"Test Case {i+1}: \"{test_case.query[:50]}...\"")
        print(f"  Query frequency: {query_freq:.2f} ({query_level.name})")
        print(f"  Expected level:  {test_case.query_level.name}")
        
        # Document frequencies
        print(f"  Document frequencies:")
        for j, doc in enumerate(test_case.doc_texts):
            doc_freq = freq_estimator.estimate(doc)
            doc_level = freq_estimator.estimate_level(doc)
            actual_level = test_case.doc_levels[j]
            match = "✓" if doc_level == actual_level else "✗"
            print(f"    Doc {j+1}: {doc_freq:.2f} ({doc_level.name:<15}) "
                  f"actual: {actual_level.name:<15} {match}")
        
        # Evaluate methods
        metrics_no_freq, ranking_no = evaluate_frequency_retrieval(
            test_case, use_frequency=False, vectorizer=vectorizer
        )
        results['no_freq'].append(metrics_no_freq)
        
        metrics_weighted, ranking_w = evaluate_frequency_retrieval(
            test_case, use_frequency=True, frequency_mode='weighted', 
            alpha=0.3, vectorizer=vectorizer
        )
        results['weighted'].append(metrics_weighted)
        
        metrics_filtered, ranking_f = evaluate_frequency_retrieval(
            test_case, use_frequency=True, frequency_mode='filtered',
            vectorizer=vectorizer
        )
        results['filtered'].append(metrics_filtered)
        
        print(f"\n  Results:")
        print(f"    No Frequency:  Level@1={metrics_no_freq.level_match_at_1}, "
              f"Level@3={metrics_no_freq.level_match_at_3}, "
              f"BestRank={metrics_no_freq.best_level_rank}")
        print(f"    Weighted:      Level@1={metrics_weighted.level_match_at_1}, "
              f"Level@3={metrics_weighted.level_match_at_3}, "
              f"BestRank={metrics_weighted.best_level_rank}")
        print(f"    Filtered:      Level@1={metrics_filtered.level_match_at_1}, "
              f"Level@3={metrics_filtered.level_match_at_3}, "
              f"BestRank={metrics_filtered.best_level_rank}")
        print()
    
    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    for method, metrics_list in results.items():
        level_at_1 = sum(1 for m in metrics_list if m.level_match_at_1)
        level_at_3 = sum(1 for m in metrics_list if m.level_match_at_3)
        avg_best_rank = np.mean([m.best_level_rank for m in metrics_list])
        avg_freq_diff = np.mean([m.top_freq_diff for m in metrics_list])
        
        print(f"\n{method.upper()}")
        print(f"  Level match @1:    {level_at_1}/{len(metrics_list)}")
        print(f"  Level match @3:    {level_at_3}/{len(metrics_list)}")
        print(f"  Avg best rank:     {avg_best_rank:.2f}")
        print(f"  Avg freq diff @1:  {avg_freq_diff:.3f}")
    
    # Analysis
    print("\n" + "-" * 70)
    print("FREQUENCY ESTIMATION ANALYSIS")
    print("-" * 70)
    
    # How well does the estimator predict actual levels?
    correct = 0
    total = 0
    
    for test_case in test_cases:
        for doc, actual_level in zip(test_case.doc_texts, test_case.doc_levels):
            predicted = freq_estimator.estimate_level(doc)
            if predicted == actual_level:
                correct += 1
            total += 1
    
    print(f"\nFrequency estimator accuracy: {correct}/{total} = {correct/total:.1%}")
    
    # Conclusions
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    no_freq_at_1 = sum(1 for m in results['no_freq'] if m.level_match_at_1)
    weighted_at_1 = sum(1 for m in results['weighted'] if m.level_match_at_1)
    filtered_at_1 = sum(1 for m in results['filtered'] if m.level_match_at_1)
    
    improvement = weighted_at_1 - no_freq_at_1
    
    if improvement > 0:
        print(f"\nPOSITIVE: Frequency weighting improved Level@1 by {improvement}")
    elif improvement == 0:
        print(f"\nNEUTRAL: Frequency weighting had no effect on Level@1")
    else:
        print(f"\nNEGATIVE: Frequency weighting hurt Level@1 by {-improvement}")
    
    print("\nDISCUSSION:")
    print("  - Frequency estimation is inherently noisy")
    print("  - Abstraction level is subjective and continuous")
    print("  - TF-IDF may already capture some abstraction info")
    print("  - May need learned frequency predictor")


def analyze_frequency_estimator():
    """Detailed analysis of the frequency estimator"""
    
    print("\n" + "=" * 70)
    print("FREQUENCY ESTIMATOR DEEP DIVE")
    print("=" * 70)
    
    estimator = FrequencyEstimator()
    
    # Test sentences across abstraction spectrum
    test_sentences = [
        # Very abstract
        ("Machine learning is a paradigm of artificial intelligence.", 
         AbstractionLevel.VERY_ABSTRACT),
        ("The concept of recursion represents a fundamental programming principle.",
         AbstractionLevel.VERY_ABSTRACT),
        
        # Abstract
        ("Neural networks learn by adjusting weights through backpropagation.",
         AbstractionLevel.ABSTRACT),
        ("Databases organize information for efficient retrieval.",
         AbstractionLevel.ABSTRACT),
        
        # Mixed
        ("To train a model, split data 80/20 and iterate until convergence.",
         AbstractionLevel.MIXED),
        ("The algorithm uses gradient descent with learning rate tuning.",
         AbstractionLevel.MIXED),
        
        # Specific
        ("Set learning_rate=0.001 and batch_size=32 for optimal results.",
         AbstractionLevel.SPECIFIC),
        ("The experiment achieved 95.3% accuracy after 100 epochs.",
         AbstractionLevel.SPECIFIC),
        
        # Very specific
        ("model.fit(X_train, y_train, epochs=100, batch_size=32)",
         AbstractionLevel.VERY_SPECIFIC),
        ("SELECT * FROM users WHERE created_at > '2024-01-15';",
         AbstractionLevel.VERY_SPECIFIC),
    ]
    
    print(f"\n{'Text':<60} {'Est':>6} {'Pred':<15} {'Actual':<15} {'Match':>5}")
    print("-" * 105)
    
    correct = 0
    for text, actual_level in test_sentences:
        freq = estimator.estimate(text)
        pred_level = estimator.estimate_level(text)
        match = "✓" if pred_level == actual_level else "✗"
        if pred_level == actual_level:
            correct += 1
        
        text_short = text[:57] + "..." if len(text) > 60 else text
        print(f"{text_short:<60} {freq:>6.2f} {pred_level.name:<15} {actual_level.name:<15} {match:>5}")
    
    print(f"\nAccuracy: {correct}/{len(test_sentences)} = {correct/len(test_sentences):.1%}")


if __name__ == "__main__":
    run_experiment()
    analyze_frequency_estimator()
