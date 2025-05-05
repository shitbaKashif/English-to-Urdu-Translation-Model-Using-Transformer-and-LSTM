from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

def compute_bleu(references, hypotheses):
    refs = [[ref.split()] for ref in references]
    hyps = [hyp.split() for hyp in hypotheses]
    score = corpus_bleu(refs, hyps)
    return score

def compute_rouge(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    return scores
