import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random


def load_json_lenient(path: Path) -> List[Dict[str, Any]]:
    """
    Load JSON from a file in a lenient way:
    - If it's a valid JSON list -> return list
    - If it's a valid JSON object (dict) with numeric keys -> return list of values
    - If invalid JSON -> scan for balanced JSON objects and parse those
    Returns a list of objects (dicts). Silently skips objects that fail to parse.
    """
    text = path.read_text(encoding="utf-8")

    # First, try strict parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [obj for obj in data if isinstance(obj, dict)]
        if isinstance(data, dict):
            # Could be a mapping of ids to objects
            values = list(data.values())
            return [obj for obj in values if isinstance(obj, dict)]
    except Exception:
        pass

    # If not strict, try to salvage by scanning for top-level JSON objects
    objects: List[Dict[str, Any]] = []
    in_string = False
    escape = False
    depth = 0
    start_idx: Optional[int] = None

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        # not in string
        if ch == '"':
            in_string = True
            continue
        if ch == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    candidate = text[start_idx : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except Exception:
                        pass
                    start_idx = None

    return objects


def normalize_record(obj: Dict[str, Any], next_id: int) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Normalize different schemas into a unified record:
    - pt legal-doc schema: arquivo/contexto/pergunta/resposta
    - en chat schema: question/answer/evidence/type

        Output schema (match qa_dataset.json):
        {
            "arquivo": str,
            "contexto": str,
            "pergunta": str,
            "resposta": str,
        }
    """
    # If a type field exists, only accept when type == "Single"
    if "type" in obj and obj.get("type") != "Single":
        return None, next_id

    # Portuguese schema
    if {"arquivo", "contexto", "pergunta", "resposta"}.issubset(obj.keys()):
        pergunta = obj.get("pergunta")
        resposta = obj.get("resposta")
        if not isinstance(pergunta, str) or not isinstance(resposta, str):
            return None, next_id
        arquivo_val = (obj.get("arquivo") or "")
        if isinstance(arquivo_val, str):
            arquivo_val = arquivo_val.replace(":", "")
        rec = {
            "arquivo": arquivo_val,
            "contexto": obj.get("contexto") or "",
            "pergunta": pergunta,
            "resposta": resposta,
        }
        return rec, next_id + 1

    # English schema
    if {"question", "answer"}.issubset(obj.keys()):
        question = obj.get("question")
        answer = obj.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            return None, next_id
        # Per user: evidence is the same as source -> map to "arquivo" in target schema
        arquivo = obj.get("evidence") or ""
        if isinstance(arquivo, str):
            arquivo = arquivo.replace(":", "")
        rec = {
            "arquivo": arquivo,
            "contexto": "",
            "pergunta": question,
            "resposta": answer,
        }
        return rec, next_id + 1

    # Unknown schema - skip
    return None, next_id


def dedupe(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for r in records:
        q = (r.get("pergunta") or "").strip().lower()
        a = (r.get("resposta") or "").strip().lower()
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def merge(
    paths: List[Path],
    out_path: Path,
    per_file_limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    raw_objects: List[Dict[str, Any]] = []
    sampled_objects: List[Dict[str, Any]] = []
    rng = random.Random(seed) if seed is not None else random
    per_file_counts: List[Dict[str, int]] = []

    for p in paths:
        objs = load_json_lenient(p)
        raw_objects.extend(objs)
        if per_file_limit is not None and per_file_limit >= 0:
            if len(objs) > per_file_limit:
                # Randomly sample without replacement
                try:
                    chosen = rng.sample(objs, per_file_limit)
                except ValueError:
                    # If per_file_limit > len(objs), fallback to all
                    chosen = objs
            else:
                chosen = objs
        else:
            chosen = objs
        sampled_objects.extend(chosen)
        per_file_counts.append({"total": len(objs), "used": len(chosen)})

    # Normalize
    normalized: List[Dict[str, Any]] = []
    nid = 1
    for obj in sampled_objects:
        rec, nid = normalize_record(obj, nid)
        if rec is not None:
            normalized.append(rec)

    # Dedupe by (question, answer)
    deduped = dedupe(normalized)

    # Write in the same format as qa_dataset.json (list of dicts with pt keys)
    out_path.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "input_files": [str(p) for p in paths],
        "raw_count": len(raw_objects),
        "per_file_counts": per_file_counts,
        "sampled_count": len(sampled_objects),
        "limit_per_file": per_file_limit,
        "seed": seed,
        "normalized_count": len(normalized),
        "deduped_count": len(deduped),
        "output_file": str(out_path),
    }


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(
            "Usage: python merge_datasets.py <input1.json> <input2.json> [more ...] -o <output.json>",
            file=sys.stderr,
        )
        # Default to known files if present in datasets folder
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        datasets_dir = project_root / "datasets"
        
        candidate1 = datasets_dir / "qa_dataset.json"
        candidate2 = datasets_dir / "mixed_qa_dataset.json"
        if candidate1.exists() and candidate2.exists():
            inputs = [candidate1, candidate2]
            out = datasets_dir / "merged_qa_dataset.json"
            stats = merge(inputs, out)
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            print(f"Merged into {out}")
            return 0
        return 2

    # Parse args
    paths: List[Path] = []
    out: Optional[Path] = None
    per_file_limit: Optional[int] = None
    seed: Optional[int] = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "-o" and i + 1 < len(argv):
            out = Path(argv[i + 1])
            i += 2
            continue
        if (arg == "-n" or arg == "--limit-per-file") and i + 1 < len(argv):
            try:
                per_file_limit = int(argv[i + 1])
            except ValueError:
                per_file_limit = None
            i += 2
            continue
        if arg == "--seed" and i + 1 < len(argv):
            try:
                seed = int(argv[i + 1])
            except ValueError:
                seed = None
            i += 2
            continue
        paths.append(Path(arg))
        i += 1

    if out is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        datasets_dir = project_root / "datasets"
        out = datasets_dir / "merged_qa_dataset.json"

    stats = merge(paths, out, per_file_limit=per_file_limit, seed=seed)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Merged into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
