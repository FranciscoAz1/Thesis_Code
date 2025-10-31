"""
compare_agent_answers.py

Usage:
  python compare_agent_answers.py \
    --agent_csv Agent_Answers.csv \
    --qa_json qa_dataset.json \
    --out comparison_report.csv

This script pairs agent outputs (from a semicolon-separated CSV) with QA entries (JSON), matches by filename, and computes
simple metrics: exact, substring, token_recall, and context_jaccard. Writes results to CSV.

Optional: pass --use_bert to compute a BERT cosine similarity (requires sentence-transformers).
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Optional NLP dependencies; load lazily where needed
try:
    from nltk.metrics import edit_distance
    from nltk.tokenize import sent_tokenize
except Exception:
    def edit_distance(a, b):
        raise RuntimeError('nltk.edit_distance not available; install nltk')
    def sent_tokenize(s):
        return [s]

try:
    from rouge import Rouge
except Exception:
    Rouge = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


TOKEN_SPLIT_RE = re.compile(r"\W+", re.UNICODE)

def normalize(s: str) -> str:
    return TOKEN_SPLIT_RE.sub(" ", (s or '').lower()).strip()


def token_set(s: str) -> set:
    return {t for t in normalize(s).split() if t}


def _lazy_rouge():
    if Rouge is None:
        raise RuntimeError('rouge package not installed')
    return Rouge()

_BERT_MODEL = None
def _lazy_bert():
    global _BERT_MODEL
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers not installed')
    if _BERT_MODEL is None:
        _BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _BERT_MODEL


def calculate_best_similarity(sentences: list[str], target: str, method="levenshtein", n=1):
    # filter out empty sentences and ensure target is non-empty
    sentences = [s for s in (sentences or []) if s and s.strip()]
    if not sentences or not (target or '').strip():
        return 0.0
    tgt_tokens = (target or '').lower().split()
    scores = []
    if method == "jaccard":
        tgt_set = set(tgt_tokens)
        for s in sentences:
            s_tokens = set(s.lower().split())
            inter = set(s_tokens) & tgt_set
            union = set(s_tokens) | tgt_set
            scores.append(len(inter) / len(union) if union else 0.0)
    elif method == "levenshtein":
        tgt_len = max(len(tgt_tokens), 1)
        for s in sentences:
            dist = edit_distance(tgt_tokens, s.lower().split())
            norm = max(tgt_len, len(s.split()))
            scores.append(1 - dist / norm if norm else 0.0)
    elif method == "rouge":
        key = f"rouge-{n}"
        r_inst = _lazy_rouge()
        for s in sentences:
            # rouge lib raises ValueError on empty hypothesis/reference; guard
            try:
                r = r_inst.get_scores(s, target)
            except ValueError:
                scores.append(0.0)
                continue
            if not r:
                scores.append(0.0)
            else:
                scores.append(r[0].get(key, {}).get("f", 0.0))
    elif method == "bert":
        model = _lazy_bert()
        embeddings = model.encode(sentences + [target], show_progress_bar=False)
        tgt_vec = embeddings[-1]
        tgt_norm = np.linalg.norm(tgt_vec)
        for i in range(len(sentences)):
            v = embeddings[i]
            denom = (np.linalg.norm(v) * tgt_norm)
            scores.append(float(np.dot(v, tgt_vec) / denom) if denom else 0.0)
    elif method == "overlap":
        tgt_set = set(tgt_tokens)
        for s in sentences:
            s_set = set(s.lower().split())
            inter = s_set & tgt_set
            denom = min(len(s_set), len(tgt_set))
            scores.append(len(inter) / denom if denom else 0.0)
    elif method == "bleu":
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        _SMOOTH = SmoothingFunction().method1
        for s in sentences:
            s_tokens = s.lower().split()
            if not s_tokens or not tgt_tokens:
                scores.append(0.0)
            else:
                scores.append(sentence_bleu(
                    [tgt_tokens], s_tokens, smoothing_function=_SMOOTH, weights=tuple((1/n for _ in range(n)))
                ))
    else:
        raise ValueError("Unsupported method.")
    return max(scores) if scores else 0.0


def compute_similarity(answer: str, gold: str, use_bert: bool = True) -> dict:
    sentences = sent_tokenize(answer)
    return {
        'jaccard': calculate_best_similarity(sentences, gold, method="jaccard"),
        'rouge1_f': calculate_best_similarity(sentences, gold, method="rouge", n=1),
        'overlap': calculate_best_similarity(sentences, gold, method="overlap"),
        'bleu': calculate_best_similarity(sentences, gold, method="bleu"),
        'bert_cos': calculate_best_similarity(sentences, gold, method="bert") if use_bert else None,
    }


def compute_metrics(answer: str, gold: str, use_bert: bool = True) -> dict:
    a_norm, g_norm = normalize(answer), normalize(gold)
    exact_m = bool(g_norm) and a_norm == g_norm
    substring_m = bool(g_norm) and g_norm in a_norm
    ts_a, ts_g = token_set(answer), token_set(gold)
    token_recall_m = (len(ts_a & ts_g) / len(ts_g)) if ts_g else 0.0
    sim = compute_similarity(answer, gold, use_bert=use_bert)
    if sim.get('bert_cos') is None:
        sim.pop('bert_cos', None)
    return {'exact': exact_m, 'substring': substring_m, 'token_recall': token_recall_m, **sim}


def exact(a: str, g: str) -> bool:
    return normalize(a) == normalize(g) and bool(normalize(g))


def substring(a: str, g: str) -> bool:
    gn = normalize(g)
    return bool(gn) and (gn in normalize(a))


def token_recall(a: str, g: str) -> float:
    gs = token_set(g)
    if not gs:
        return 0.0
    as_ = token_set(a)
    return len(as_ & gs) / len(gs)


def jaccard(a: str, g: str) -> float:
    A = token_set(a); B = token_set(g)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def parse_agent_csv(path: Path) -> list:
    """Parse semicolon-separated CSV and return ordered list of (file_field, answer) tuples.
    Keeps original CSV order to allow positional pairing with QA list.
    """
    text = path.read_text(encoding='utf-8', errors='ignore')
    lines = [l for l in text.splitlines() if l.strip()]
    reader = csv.DictReader(lines, delimiter=';')
    entries = []
    for r in reader:
        # heuristics to find filename column
        file_field = None
        for k in r.keys():
            if not k:
                continue
            if any(x in k.lower() for x in ('doc', 'arquivo', 'file', 'doc cont')):
                file_field = r[k]
                break
        if not file_field:
            # search values for pattern like documento_123_xxx
            m = re.search(r'documento_\d+_[\w\.\-]+', '\t'.join([v or '' for v in r.values()]))
            file_field = m.group(0) if m else ''
        # try common answer column names
        answer = ''
        for cand in ('Answer', 'answer', 'Resposta', 'Resposta\n'):
            if cand in r and r[cand] is not None:
                answer = r[cand]
                break
        if not answer:
            vals = list(r.values())
            answer = vals[-1] if vals else ''
        entries.append((file_field.strip() if file_field else '', (answer or '').strip()))
    return entries



def load_agent_csv(path: Path) -> list:
    """Load agent answers and gold answers from a CSV (e.g., benchmark_results_openai.csv)."""
    rows = []
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Expect columns: question, gold_answer, expected_file, expected_context, final_answer
            rows.append({
                'question': r.get('question', ''),
                'gold_answer': r.get('gold_answer', ''),
                'expected_file': r.get('expected_file', ''),
                'expected_context': r.get('expected_context', ''),
                'agent_answer': r.get('final_answer', ''),
            })
    return rows



def compare_from_csv_rows(rows: list):
    """Compare agent answers and gold answers from a single CSV row list."""
    out_rows = []
    for r in rows:
        agent_answer = r.get('agent_answer', '')
        gold = r.get('gold_answer', '')
        expected_file = r.get('expected_file', '')
        # Check if expected_file is mentioned in the agent answer
        document_found = expected_file and expected_file in agent_answer
        metrics = compute_metrics(agent_answer, gold, use_bert=USE_BERT_FLAG)
        out_row = {
            'expected_file': expected_file,
            'agent_answer': agent_answer,
            'gold': gold,
            'document_found': document_found,
            **metrics
        }
        out_rows.append(out_row)
    return out_rows


def write_csv(out: Path, rows: list):
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in w.fieldnames})


def summarize(rows: list) -> dict:
    total = len(rows)
    # consider an agent answer "found" when agent_answer is non-empty
    found = sum(1 for r in rows if r.get('agent_answer'))
    document_matches = sum(1 for r in rows if r.get('document_found'))
    exacts = sum(1 for r in rows if r.get('exact'))
    subs = sum(1 for r in rows if r.get('substring'))

    def mean(values):
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    avg_token_recall = mean([r.get('token_recall', 0.0) for r in rows if r.get('agent_answer')])
    avg_jaccard = mean([r.get('jaccard', 0.0) for r in rows if r.get('agent_answer')])
    avg_rouge1 = mean([r.get('rouge1_f', 0.0) for r in rows if r.get('agent_answer')])
    avg_overlap = mean([r.get('overlap', 0.0) for r in rows if r.get('agent_answer')])
    avg_bleu = mean([r.get('bleu', 0.0) for r in rows if r.get('agent_answer')])
    bert_vals = [r.get('bert_cos') for r in rows if r.get('agent_answer') and r.get('bert_cos') is not None]
    avg_bert = mean(bert_vals)

    return {
        'total': total,
        'found': found,
        'document_matches': document_matches,
        'exact': exacts,
        'substring': subs,
        'avg_token_recall': avg_token_recall,
        'avg_jaccard': avg_jaccard,
        'avg_rouge1_f': avg_rouge1,
        'avg_overlap': avg_overlap,
        'avg_bleu': avg_bleu,
        'avg_bert_cos': avg_bert,
    }



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent_csv', required=False, default='results/benchmark_results_openai.csv')
    p.add_argument('--out', default='results/comparison_report.csv')
    p.add_argument('--use_bert', action='store_true', help='Compute BERT similarity (requires sentence-transformers)', default=True)
    args = p.parse_args()

    agent_path = Path(args.agent_csv)
    out_path = Path(args.out)

    if not agent_path.exists():
        print(f"Agent CSV not found: {agent_path}")
        return

    # expose flag to comparison scope
    global USE_BERT_FLAG
    USE_BERT_FLAG = bool(args.use_bert)
    csv_rows = load_agent_csv(agent_path)
    rows = compare_from_csv_rows(csv_rows)
    write_csv(out_path, rows)
    s = summarize(rows)
    print("SUMMARY:", s)
    print(f"Results written to: {out_path}")


if __name__ == '__main__':
    main()
