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

from sqlalchemy import true
from legal_document_agent_openai import get_agent_and_session, openai_agent, run_agent
from agents import Agent, Runner, SQLiteSession, mcp

# Path to your QA dataset (JSON format) - using relative paths
QA_JSON_PATH = PROJECT_ROOT / "datasets" / "merged_qa_dataset.json"
OUTPUT_CSV_PATH = PROJECT_ROOT / "results" / "benchmark_results_openai.csv"

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
    parser = argparse.ArgumentParser(description="Benchmark legal document agent (OpenAI) with a QA dataset.")
    parser.add_argument("--index", type=int, default=None, help="Run only the question at this 1-based index.")
    parser.add_argument("--contains", type=str, default=None, help="Run only questions containing this substring (case-insensitive).")
    parser.add_argument("--start_index", type=int, default=None, help="Start from this 1-based index after filtering (inclusive). Ignored if --index is provided.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging during agent run.")
    parser.add_argument("--retries", type=int, default=5, help="Number of retry attempts if the agent call fails.")
    parser.add_argument("--retry_delay", type=float, default=2.0, help="Delay in seconds between retry attempts.")
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
    elif args.start_index is not None:
        # 1-based start index; guard bounds
        if args.start_index < 1 or args.start_index > len(selected):
            raise IndexError(f"--start_index {args.start_index} out of range (1..{len(selected)}) after filtering.")
        selected = selected[args.start_index - 1:]

    async def _try_openai_agent_with_retries(agent, session, question: str, verbose: bool, retries: int, delay: float):
        """Try calling openai_agent with retries. Returns result or None if all attempts fail."""
        attempts = retries + 1  # first try + retries
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                return await run_agent(agent, session, question, verbose)
            except Exception as e:
                last_error = e
                print(f"Agent error on attempt {attempt}/{attempts}: {e}")
                if attempt < attempts:
                    print(f"Retrying in {delay} seconds...")
                    try:
                        await asyncio.sleep(delay)
                    except Exception:
                        pass
        print(f"All {attempts} attempts failed for question: {question}")
        return None
    
    # Initialize agent once outside the loop
    agent, session = await get_agent_and_session(verbose=args.verbose)

    with open(OUTPUT_CSV_PATH, 'a', encoding='utf-8', newline='') as f:
        fieldnames = [
            "question", "gold_answer", "expected_file", "expected_context", "final_answer"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Only write header if the file is new
        if f.tell() == 0:
            writer.writeheader()
        for i, qa in enumerate(selected, 1):
            print(f"\n{'='*80}\nQ{i}: {qa['question']}")
            result = await _try_openai_agent_with_retries(
                agent, session,
                qa['question'], args.verbose, args.retries, args.retry_delay
            )
            if result is None:
                # Could not get a result after retries; skip
                continue
            row = {
                "question": qa['question'],
                "gold_answer": qa['gold_answer'],
                "expected_file": qa['file'],
                "expected_context": qa['context'],
                "final_answer": getattr(result, 'final_output', ''),
            }
            rows.append(row)
            print(f"Final Answer: {row['final_answer']}")
            writer.writerow(row)
            # Flush buffers and ensure the row is persisted to disk immediately
            f.flush()
            os.fsync(f.fileno())
            await session.clear_session()

    print(f"\nBenchmark complete. Results saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(benchmark_agent(args))






