# %%
# Cell 0: RAG Initialization (Run First)
# -------------------------------------
# Loads embedding model, builds embedding_func, and instantiates a MiniRAG object.
# Does NOT ingest documents. Use the next cell to index.

import os, torch, sys
import minirag
from transformers import AutoTokenizer, AutoModel
from minirag.llm.hf import hf_embed
from minirag.utils import EmbeddingFunc
from minirag.llm import ollama
from minirag import MiniRAG
from tqdm.auto import tqdm
import asyncio
import os, time, json, random, gc, asyncio
from pathlib import Path
import psutil, torch
import minirag
from minirag.llm import ollama
from minirag.utils import EmbeddingFunc
from transformers import AutoTokenizer, AutoModel
from minirag.llm.hf import hf_embed
import os, csv, time, json, random, re, statistics, asyncio, math
from pathlib import Path
# from minirag import QueryParam
from minirag.utils import calculate_similarity  # legacy helper (returns indices) – not used now

# Extra metric libs (lazy loads handled in compute_similarity)
from nltk.metrics import edit_distance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from rouge import Rouge
from sentence_transformers import SentenceTransformer
import numpy as np
sys.path.append(r'c:\Users\Francisco Azeredo\OneDrive\Documents\tecnico\5 ano\tese\Código\Chatbot\lightrag')
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete
from lightrag.kg.shared_storage import initialize_pipeline_status 
from lightrag import QueryParam
def main():
    # Core configuration (shared by later cells)
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    WORKING_DIR = r"C:\\Users\\Francisco Azeredo\\OneDrive\\Documents\\tecnico\\5 ano\\tese\\Código\\MiniRAG\\notebooks\\storage"
    LLM_MODEL_NAME = "qwen2m:latest"  # set to None if no local Ollama model
    LOG_LEVEL = "CRITICAL"

    os.makedirs(WORKING_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Init device:", device)

    print("Loading embedding tokenizer/model...")
    _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    _embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device)
    _embed_model.eval()

    async def _embed_batch(texts: list[str]):
        return await hf_embed(texts, tokenizer=_tokenizer, embed_model=_embed_model)

    async def _embed_dispatch(input_text):
        if isinstance(input_text, str):
            return (await _embed_batch([input_text]))[0]
        if isinstance(input_text, (list, tuple)) and all(isinstance(t, str) for t in input_text):
            return await _embed_batch(list(input_text))
        raise TypeError(f"Unsupported input type for embedding_func: {type(input_text)}")

    _embedding_func = EmbeddingFunc(
        embedding_dim=_embed_model.config.hidden_size,
        max_token_size=_tokenizer.model_max_length,
        # func=_embed_dispatch,
        func = lambda texts: hf_embed(texts, tokenizer=_tokenizer, embed_model=_embed_model),
    )

    # rag = minirag.MiniRAG(
    #     working_dir=WORKING_DIR,
    #     llm_model_func=ollama.ollama_model_complete if LLM_MODEL_NAME else None,
    #     llm_model_name=LLM_MODEL_NAME,
    #     embedding_func=_embedding_func,
    #     log_level=LOG_LEVEL,
    #     suppress_httpx_logging=True
    # )
    async def initialize_rag():
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=ollama.ollama_model_complete if LLM_MODEL_NAME else None,
            llm_model_name=LLM_MODEL_NAME,
            embedding_func=_embedding_func,
            log_level=LOG_LEVEL,
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()
        return rag

    rag = asyncio.run(initialize_rag())

    print("RAG ready. Proceed to Cell 1 to ingest documents.")


    """
    Cell 1: Document Ingestion / Indexing Only
    -----------------------------------------
    Run this FIRST. It builds the MiniRAG index (vectors + KG) from source documents.
    No query / evaluation logic here.
    """
    # ---------------- User Config ----------------
    RANDOM_SEED = 42
    SHUFFLE_DOCS = False
    MAX_DOCS = None  # set int to limit docs
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DATASET_DIR = r"C:\\Users\\Francisco Azeredo\\OneDrive\\Documents\\tecnico\\5 ano\\tese\\Código\\MiniRAG\\dataset\\LiHua-World\\data\\"
    WORKING_DIR = r"C:\\Users\\Francisco Azeredo\\OneDrive\\Documents\\tecnico\\5 ano\\tese\\Código\\MiniRAG\\notebooks\\storage"
    LLM_MODEL_NAME = "qwen2m:latest"  # set to None if no LLM available
    LOG_LEVEL = "CRITICAL"

    random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    try:
        PROCESS = psutil.Process()
    except Exception:
        PROCESS = None

    # ---------------- Helpers ----------------

    def memory_mb():
        if PROCESS is None: return None
        return PROCESS.memory_info().rss / (1024 * 1024)

    def read_text_from_file(path: Path) -> str:
        suffix = path.suffix.lower()
        try:
            if suffix in {".txt", ".md"}:
                return path.read_text(encoding="utf-8", errors="ignore")
            if suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                for k in ("text","content","body","article"):
                    if isinstance(data, dict) and k in data and isinstance(data[k], str):
                        return data[k]
                return json.dumps(data)
            if suffix in {".jsonl", ".ndjson"}:
                lines = []
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line=line.strip()
                        if not line: continue
                        try:
                            obj=json.loads(line)
                            if isinstance(obj, dict):
                                for k in ("text","content","body","article"):
                                    if k in obj and isinstance(obj[k], str):
                                        lines.append(obj[k]); break
                                else:
                                    lines.append(json.dumps(obj))
                            else:
                                lines.append(str(obj))
                        except Exception:
                            lines.append(line)
                return "\n".join(lines)
        except Exception as e:
            return f"ERROR_READING_FILE {path.name}: {e}"
        return ""

    def load_documents(root_dir: str):
        exts = (".txt", ".md", ".json", ".jsonl", ".ndjson")
        paths = [p for p in Path(root_dir).rglob("*") if p.suffix.lower() in exts and p.is_file()]
        if SHUFFLE_DOCS: random.shuffle(paths)
        docs = []
        for p in paths[:2]:
            if MAX_DOCS and len(docs) >= MAX_DOCS: break
            text = read_text_from_file(p).strip()
            if not text: continue
            docs.append({"id": f"doc_{len(docs)}", "text": text, "source_path": str(p)})
        return docs

    # ---------------- Indexing ----------------
    def index_documents():
        print("Loading documents...")
        t0 = time.perf_counter(); docs = load_documents(DATASET_DIR)
        print(f"Loaded {len(docs)} docs in {time.perf_counter()-t0:.2f}s")
        if not docs:
            print("No documents found; adjust DATASET_DIR."); return
        start_mem = memory_mb()
        if start_mem is not None: print(f"Start RSS: {start_mem:.2f} MB")
        texts = [d['text'] for d in docs]
        metas = [{"id": d['id'], "source": d['source_path']} for d in docs]
        print("Indexing with ainsert() ...")
        t1 = time.perf_counter()
        for d in tqdm(docs, desc="Indexing docs", unit="doc"):
            if 'text' not in d or not d['text'].strip():
                print(f"Skipping empty doc {d.get('id')}")
            try:
                rag.insert(input=d['text'], ids=d['id'], file_paths=d['source_path'])
            except Exception as batch_e:
                print(f"Batch insert failed: {batch_e}; fallback per-doc")
        dur = time.perf_counter()-t1
        print(f"Inserted {len(texts)} docs in {dur:.2f}s ({len(texts)/dur:.2f} docs/s)")
        gc.collect(); end_mem = memory_mb()
        if end_mem is not None: print(f"End RSS: {end_mem:.2f} MB (Δ {end_mem - start_mem:.2f} MB)")

    # index_documents()
    print("Indexing complete. Proceed to Cell 2 for querying & evaluation.")

    # %%
    # Cell 2: Query & QA Evaluation
    # ----------------------------------------------
    # Run AFTER Cell 1. Uses the global `rag` object and indexed data.
    # Supports:
    #  - Loading LiHua-World QA pairs from query_set.csv
    #  - Evaluating answer quality with simple + lexical + semantic metrics
    #  - Optional CSV logging



    # -------- Configuration --------
    QA_CSV_PATH = r"C:\Users\Francisco Azeredo\OneDrive\Documents\tecnico\5 ano\tese\Código\MiniRAG\dataset\LiHua-World\qa\query_set.csv"
    OUTPUT_CSV_PATH = r"C:\Users\Francisco Azeredo\OneDrive\Documents\tecnico\5 ano\tese\Código\MiniRAG\notebooks"  # set to None to skip saving
    QUERY_MODE = "naive"      # mini | light | naive | doc | meta | bm25
    TOP_K = 5
    MAX_Q = None             # limit question count
    RANDOM_SEED = 42
    USE_BERT_SIM = True       # toggle semantic similarity (slower)
    random.seed(RANDOM_SEED)

    # -------- Metrics Helpers --------
    TOKEN_SPLIT_RE = re.compile(r"\W+", re.UNICODE)

    # lazy globals
    _ROUGE = None
    _BERT_MODEL = None
    _SMOOTH_FN = SmoothingFunction().method1


    def _lazy_rouge():
        global _ROUGE
        if _ROUGE is None:
            _ROUGE = Rouge()
        return _ROUGE


    def _lazy_bert():
        global _BERT_MODEL
        if _BERT_MODEL is None:
            _BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        return _BERT_MODEL


    def normalize_text(s: str) -> str:
        try:
            return TOKEN_SPLIT_RE.sub(" ", s.lower()).strip()
        except AttributeError:
            return normalize_text(s[0])
        except Exception as e:
            print(f"Error normalizing text: {e}")
            return s

    def token_set(s: str) -> set[str]:
        return {t for t in normalize_text(s).split() if t}

    _BERT_MODEL = None
    _ROUGE = None
    _SMOOTH = SmoothingFunction().method1

    def calculate_best_similarity(sentences: list[str], target: str, method="levenshtein", n=1):
        """
        Returns the highest similarity score (float) between any sentence in `sentences` and `target`.
        Methods: jaccard | levenshtein | rouge | bert | overlap | bleu
        For rouge, n=1 or 2 selects rouge-1 or rouge-2 F.
        """
        if not sentences:
            return 0.0
        tgt_tokens = target.lower().split()
        scores = []

        if method == "jaccard":
            tgt_set = set(tgt_tokens)
            for s in sentences:
                s_tokens = set(s.lower().split())
                inter = set(s_tokens).intersection(set(tgt_set))
                union = set(s_tokens).union(set(tgt_set))
                scores.append(len(inter) / len(union) if union else 0.0)

        elif method == "levenshtein":
            tgt_len = max(len(tgt_tokens), 1)
            for s in sentences:
                dist = edit_distance(tgt_tokens, s.lower().split())
                norm = max(tgt_len, len(s.split()))
                scores.append(1 - dist / norm if norm else 0.0)

        elif method == "rouge":
            _ROUGE = Rouge()
            key = f"rouge-{n}"
            for s in sentences:
                r = _ROUGE.get_scores(s, target)
                scores.append(r[0].get(key, {}).get("f", 0.0))

        elif method == "bert":
            _BERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = _BERT_MODEL.encode(sentences + [target], show_progress_bar=False)
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
            tgt_bleu = word_tokenize(target.lower())
            for s in sentences:
                s_bleu = word_tokenize(s.lower())
                scores.append(sentence_bleu([tgt_bleu], s_bleu, smoothing_function=_SMOOTH))
        else:
            raise ValueError("Unsupported method.")

        return max(scores) if scores else 0.0

    def compute_similarity(answer: str, gold: str, use_bert: bool = True) -> dict:
        """Compute a bundle of similarity scores between answer and gold.

        Returns keys:
        jaccard, levenshtein, rouge1_f, rouge2_f, overlap, bleu, bert_cos (optional)
        """
        sentences = sent_tokenize(answer)
        jaccard = calculate_best_similarity(sentences, gold, method="jaccard")
        levenshtein = calculate_best_similarity(sentences, gold, method="levenshtein")
        rouge1_f = calculate_best_similarity(sentences, gold, method="rouge", n=1)
        rouge2_f = calculate_best_similarity(sentences, gold, method="rouge", n=2)
        overlap = calculate_best_similarity(sentences, gold, method="overlap")
        bleu = calculate_best_similarity(sentences, gold, method="bleu")
        bert_cos = calculate_best_similarity(sentences, gold, method="bert") if use_bert else None

        result = {
            'jaccard': jaccard,
            'levenshtein': levenshtein,
            'rouge1_f': rouge1_f,
            'rouge2_f': rouge2_f,
            'overlap': overlap,
            'bleu': bleu,
        }
        if bert_cos is not None:
            result['bert_cos'] = bert_cos
        return result


    def compute_metrics(answer: str, gold: str) -> dict:
        # Basic lexical metrics
        a_norm, g_norm = normalize_text(answer), normalize_text(gold)
        exact = bool(g_norm) and a_norm == g_norm
        substring = bool(g_norm) and g_norm in a_norm
        ts_a, ts_g = token_set(answer), token_set(gold)
        token_recall = (len(ts_a & ts_g) / len(ts_g)) if ts_g else 0.0

        sim_bundle = compute_similarity(answer, gold, use_bert=USE_BERT_SIM)

        return {
            'exact': exact,
            'substring': substring,
            'token_recall': token_recall,
            **sim_bundle,
        }

    # -------- Load QA Pairs --------
    qa_pairs = []
    if os.path.exists(QA_CSV_PATH):
        with open(QA_CSV_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "Question" in row and "Gold Answer" in row:
                    qa_pairs.append((row["Question"].strip(), row["Gold Answer"].strip()))
    else:
        print("QA CSV not found. Provide QA_CSV_PATH or create synthetic pairs manually.")

    if MAX_Q:
        qa_pairs = qa_pairs[:MAX_Q]

    print(f"Loaded {len(qa_pairs)} QA pairs.")
    if not qa_pairs:
        raise SystemExit("No QA data available.")


    # -------- Evaluation --------
    def run_eval(mode, n):
        qp = QueryParam(mode=mode, top_k=TOP_K)
        rows = []
        latencies = []

        for i, (question, gold) in enumerate(tqdm(qa_pairs, total=len(qa_pairs), desc=f"Eval-{mode}", unit="q"), start=1):

            t0 = time.perf_counter()
            try:
                answer = rag.query(question, param=qp)
            except TypeError:
                answer = rag.aquery(question)
            latency = time.perf_counter() - t0
            latencies.append(latency)

            m = compute_metrics(answer, gold)
            row = {"question": question, "gold": gold, "answer": answer, "latency_s": latency, **m}
            rows.append(row)

            if i <= 0:
                # use tqdm.write to avoid breaking the progress bar formatting
                tqdm.write(f"Q{i}: {question[:80]}...")
                tqdm.write("Answer: " + answer[:180].replace("\n", " "))
                tqdm.write("Gold: " + gold[:180])
                # Format numeric (non-NaN) metrics to 3 decimals
                fmt_metrics = {
                    k: (f"{v:.3f}" if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else v)
                    for k, v in m.items()
                }
                tqdm.write(f"Metrics: {fmt_metrics} Latency: {latency*1000:.1f} ms")
                tqdm.write('-')
        # Aggregates
        def _avg(key):
            vals = [r[key] for r in rows if key in r and isinstance(r[key], (int,float))]
            return sum(vals)/len(vals) if vals else 0.0

        exact_rate = _avg('exact')
        substr_rate = _avg('substring')
        avg_token_recall = _avg('token_recall')
        avg_lat = sum(latencies)/len(latencies)
        p95_lat = sorted(latencies)[int(len(latencies)*0.95)-1] if len(latencies) > 1 else latencies[0]

        print(f"\nAggregate: exact={exact_rate:.2%} substring={substr_rate:.2%} token_recall={avg_token_recall:.2%}")
        for mkey in ['jaccard','levenshtein','rouge1_f','rouge2_f','overlap','bleu','bert_cos']:
            if mkey in rows[0]:
                print(f"  {mkey}: {_avg(mkey):.3f}")
        print(f"Latency: avg={avg_lat*1000:.1f} ms p95={p95_lat*1000:.1f} ms")

        os.makedirs(OUTPUT_CSV_PATH, exist_ok=True)
        OUTPUT_CSV = os.path.join(OUTPUT_CSV_PATH, f"results_{mode}{n}.csv")
        # Optional CSV
        if OUTPUT_CSV and rows:
            write_header = not os.path.exists(OUTPUT_CSV)
            with open(OUTPUT_CSV, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if write_header: writer.writeheader()
                writer.writerows(rows)
            print(f"Saved results to {OUTPUT_CSV}")
        return rows

    # Run evaluation
    eval_results1 = run_eval("naive", 5)
    eval_results2 = run_eval("hybrid", 5)
    eval_results3 = run_eval("mix", 5)
    print("Evaluation complete.")

if __name__ == "__main__":
    main()