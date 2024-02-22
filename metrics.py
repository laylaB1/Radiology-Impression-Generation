import pandas as pd
from InstructorEmbedding import INSTRUCTOR
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score

class MetricsCalculator:
    def __init__(self):
        self.model = INSTRUCTOR('hkunlp/instructor-large')
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_rouge_scores(self, reference, candidate):
        scores = self.scorer.score(reference, candidate)
        return scores

    def calculate_instructor_similarity(self, reference, candidate):
        embeddings_candidate = self.model.encode([candidate])
        embeddings_reference = self.model.encode([reference])
        similarities = cosine_similarity(embeddings_candidate, embeddings_reference)
        return similarities

    def calculate_bert_score(self, reference, candidate):
        P, R, F1 = score([candidate], [reference], lang="en")
        precision = P.mean().item()
        recall = R.mean().item()
        F1 = F1.mean().item()
        return precision, recall, F1

# Example usage:
data = pd.read_csv('path_to_your_file')
metrics_calculator = MetricsCalculator()

data[['rouge-1', 'rouge-2', 'rouge-L']] = data.apply(lambda row: metrics_calculator.calculate_rouge_scores(row['impression_last'], row['gpt_impression_stuff']), axis=1, result_type='expand')
data['IntructorSimilarity'] = data.apply(lambda row: metrics_calculator.calculate_instructor_similarity(row['impression_last'], row['gpt_impression_refine']), axis=1)
data['BERTScore'] = data.apply(lambda row: metrics_calculator.calculate_bert_score(row['impression_last'], row['re-written_impression_refine']), axis=1)

