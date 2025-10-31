import os
import json
import csv
import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from legal_document_agent import query_legal_documents

# Path to your QA dataset (JSON format) - using relative paths
QA_JSON_PATH = PROJECT_ROOT / "datasets" / "qa_dataset.json"
OUTPUT_CSV_PATH = PROJECT_ROOT / "results" / "benchmark_results.csv"

# Load QA pairs from JSON
qa_pairs = []
if os.path.exists(QA_JSON_PATH):
    with open(QA_JSON_PATH, encoding="utf-8") as f:
        qa_data = json.load(f)
        for item in qa_data:
            if "pergunta" in item and "resposta" in item:
                qa_pairs.append({
                    "question": item["pergunta"].strip(),
                    "gold_answer": item["resposta"].strip(),
                    "context": item.get("contexto", "").strip(),
                    "file": item.get("arquivo", "").strip()
                })
else:
    raise FileNotFoundError(f"QA JSON not found: {QA_JSON_PATH}")

print(f"Loaded {len(qa_pairs)} QA pairs from JSON.")

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark legal document agent with a QA dataset.")
    parser.add_argument("--index", type=int, default=None, help="Run only the question at this 1-based index.")
    parser.add_argument("--contains", type=str, default=None, help="Run only questions containing this substring (case-insensitive).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging during agent run.")
    parser.add_argument("--log-steps", action="store_true", help="Return step-by-step messages from the agent.")
    return parser.parse_args()

async def benchmark_agent(args):
    rows = []
    # Select subset based on args
    selected = qa_pairs
    if args.contains:
        needle = args.contains.lower()
        selected = [qa for qa in selected if needle in qa["question"].lower()]
    if args.index is not None:
        # 1-based index; guard bounds
        if args.index < 1 or args.index > len(selected):
            raise IndexError(f"--index {args.index} out of range (1..{len(selected)}) after filtering.")
        selected = [selected[args.index - 1]]

    for i, qa in enumerate(selected, 1):
        print(f"\n{'='*80}\nQ{i}: {qa['question']}")
        result = await query_legal_documents(qa['question'], verbose=args.verbose, log_steps=args.log_steps)
        row = {
            "question": qa['question'],
            "gold_answer": qa['gold_answer'],
            "expected_file": qa['file'],
            "expected_context": qa['context'],
            "final_answer": result.get('answer', ''),
            "success": result.get('success', False),
            "error": result.get('error', ''),
            "context_retrieved": json.dumps(result.get('context', []), ensure_ascii=False),
            "messages": json.dumps(result.get('messages', []), ensure_ascii=False) if 'messages' in result else '',
        }
        rows.append(row)
        print(f"Final Answer: {row['final_answer']}")
        print(f"Context Retrieved: {row['context_retrieved'][:300]}...")
        print(f"Messages: {row['messages'][:300]}...")
    # Write to CSV
    with open(OUTPUT_CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            "question", "gold_answer", "expected_file", "expected_context", "final_answer", "success", "error", "context_retrieved", "messages"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nBenchmark complete. Results saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(benchmark_agent(args))
