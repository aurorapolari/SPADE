"""Summary metrics, averaged over independently scored reference sets."""
from statistics import mean


def check_resources():
    from nltk.corpus import wordnet
    try:
        wordnet.ensure_loaded()
    except LookupError as error:
        raise RuntimeError("METEOR requires WordNet. Run: python -m nltk.downloader wordnet omw-1.4") from error


def score_summaries(predictions, rows):
    import sacrebleu
    from nltk.tokenize import TreebankWordTokenizer
    from nltk.translate.meteor_score import meteor_score
    from rouge_score.rouge_scorer import RougeScorer

    if len(predictions) != len(rows) or not rows:
        raise ValueError("Each prediction must correspond to one reference record")
    count = len(rows[0]["references"])
    if not count or any(len(row["references"]) != count for row in rows):
        raise ValueError("All records must have the same number of reference sets")
    check_resources()
    tokenize = TreebankWordTokenizer().tokenize
    rouge = RougeScorer(["rougeL"], use_stemmer=True)
    per_reference = []
    for index in range(count):
        references = [row["references"][index] for row in rows]
        per_reference.append({
            "bleu4": sacrebleu.corpus_bleu(predictions, [references], tokenize="13a", smooth_method="exp").score,
            "meteor": 100 * mean(meteor_score([tokenize(ref)], tokenize(pred)) for ref, pred in zip(references, predictions)),
            "rougeL": 100 * mean(rouge.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions)),
        })
    return {key: mean(item[key] for item in per_reference) for key in per_reference[0]} | {"per_reference": per_reference}
