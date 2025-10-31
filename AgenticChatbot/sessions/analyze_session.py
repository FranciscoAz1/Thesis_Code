import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


WEAVIATE_TOOLS = {
    "weaviate-origin": "origin",
    "weaviate-query": "query",
    "weaviate-follow-ref": "follow-ref",
}

# Known reference directions by property name (best-effort, used if output text parsing fails)
REF_PROP_TARGET = {
    # Etapa refs
    "aboutReports": "Ficheiro",
    "aboutEntities": "Entidade",
    "belongsToFlux": "Fluxo",
    # Fluxo refs
    "hasStages": "Etapa",
    "hasReports": "Ficheiro",
    # Ficheiro refs
    "hasEntidades": "Entidade",
    "partOfFlux": "Fluxo",
    "triggersEtapas": "Etapa",
}

TRAVERSED_RE = re.compile(r"Traversed '([^']+)'\s*->\s*([A-Za-z]+):")
JSON_BLOB_RE = re.compile(r"\{\s*\n?\s*\"data\"\s*:\s*\{", re.S)


@dataclass
class Step:
    index: int
    tool: str
    name: str
    collection: Optional[str] = None
    query: Optional[str] = None
    target_properties: Optional[List[str]] = None
    ref_prop: Optional[str] = None
    base_props: Optional[List[str]] = None
    ref_props: Optional[List[str]] = None
    traversed_to: Optional[str] = None
    traversed_items: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class GraphEdge:
    src: str
    prop: str
    dst: str
    count: int = 0


def parse_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file.
    Tries line-by-line; if unable, attempts to split concatenated JSON objects.
    """
    text = path.read_text(encoding="utf-8")
    objs: List[Dict[str, Any]] = []

    # First try: each non-empty line is a JSON object
    ok = True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        try:
            obj = json.loads(ln)
            objs.append(obj)
        except json.JSONDecodeError:
            ok = False
            break
    if ok and objs:
        for obj in objs:
            yield obj
        return

    # Second try: file might contain multiple JSON objects back-to-back
    # Split on pattern of closing+opening brace at line boundaries
    chunks: List[str] = []
    buf = []
    brace_balance = 0
    for ch in text:
        buf.append(ch)
        if ch == '{':
            brace_balance += 1
        elif ch == '}':
            brace_balance -= 1
            if brace_balance == 0:
                chunk = ''.join(buf).strip()
                chunks.append(chunk)
                buf = []
    for chunk in chunks:
        try:
            yield json.loads(chunk)
        except json.JSONDecodeError:
            # Last resort: skip badly formatted chunks
            continue


def safe_json_loads(maybe_json: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(maybe_json)
    except Exception:
        return None


def parse_output_traversal(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse the output 'Traversed' line to get (ref_prop, target_collection)."""
    m = TRAVERSED_RE.search(text)
    if m:
        return m.group(1), m.group(2)
    return None, None


def extract_target_collection_from_ref(ref_prop: Optional[str]) -> Optional[str]:
    if not ref_prop:
        return None
    return REF_PROP_TARGET.get(ref_prop)


def parse_session_items(items: List[Dict[str, Any]]) -> List[Step]:
    steps: List[Step] = []
    # Map call_id -> output text for quick lookup
    outputs: Dict[str, str] = {}

    for i, item in enumerate(items):
        t = item.get("type")
        if t == "function_call_output":
            call_id_val = item.get("call_id")
            call_id = str(call_id_val) if call_id_val is not None else None
            out = item.get("output")
            if isinstance(out, str) and call_id is not None:
                outputs[call_id] = out
            continue

    step_idx = 0
    for item in items:
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        if name not in WEAVIATE_TOOLS:
            continue
        args_raw = item.get("arguments")
        args = None
        if isinstance(args_raw, str):
            args = safe_json_loads(args_raw)
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}

        call_id_val = item.get("call_id")
        call_id = str(call_id_val) if call_id_val is not None else ""
        out_text = outputs.get(call_id, "")

        step_idx += 1
        step = Step(index=step_idx, tool=WEAVIATE_TOOLS[name], name=name)

        # Extract request details
        if args:
            step.collection = args.get("collection")
            step.query = args.get("query")
            step.target_properties = args.get("targetProperties")
            step.ref_prop = args.get("refProp")
            step.base_props = args.get("baseProps")
            step.ref_props = args.get("refProps")

        # Detect errors in output
        if "Error:" in out_text or "Query failed:" in out_text:
            step.error = out_text.strip()

        # Parse traversed info if present
        ref_prop_text, dst = parse_output_traversal(out_text)
        if ref_prop_text:
            step.ref_prop = step.ref_prop or ref_prop_text
            step.traversed_to = dst

        # If it's a direct query with JSON content, try to extract item names/titles
        if name == "weaviate-query" and out_text:
            # The output text in your logs embeds a JSON string of the GraphQL response
            # Try to locate and parse it safely
            try:
                out_json_candidate = json.loads(out_text)
                # If it's already JSON-like from the tool, dig deeper
                data = out_json_candidate.get("data", {}) if isinstance(out_json_candidate, dict) else {}
            except Exception:
                data = {}
                # Try to parse the inner quoted JSON
                try:
                    inner = json.loads(out_text).get("text")  # when tool wraps as {"type":"text","text":"{...}"}
                    if isinstance(inner, str) and '{' in inner:
                        inner_json = json.loads(inner)
                        data = inner_json.get("data", {})
                except Exception:
                    pass
            if data:
                get_block = data.get("Get", {})
                if isinstance(get_block, dict) and step.collection in get_block:
                    arr = get_block.get(step.collection) or []
                    titles = []
                    for obj in arr:
                        if isinstance(obj, dict):
                            t = obj.get("title") or obj.get("name")
                            if t:
                                titles.append(str(t))
                    step.traversed_items = titles

        steps.append(step)

    return steps


def build_graph(steps: List[Step]) -> Dict[Tuple[str, str, str], GraphEdge]:
    edges: Dict[Tuple[str, str, str], GraphEdge] = {}
    for st in steps:
        if st.ref_prop:
            src = st.collection or "?"
            dst = st.traversed_to or extract_target_collection_from_ref(st.ref_prop) or "?"
            key = (src, st.ref_prop, dst)
            if key not in edges:
                edges[key] = GraphEdge(src=src, prop=st.ref_prop, dst=dst, count=0)
            edges[key].count += 1
    return edges


def write_dot(edges: Dict[Tuple[str, str, str], GraphEdge], path: Path) -> None:
    nodes = set()
    for (src, _, dst) in edges.keys():
        if src != "?":
            nodes.add(src)
        if dst != "?":
            nodes.add(dst)
    lines = ["digraph G {"]
    for n in sorted(nodes):
        lines.append(f"  \"{n}\";")
    for e in edges.values():
        label = f"{e.prop} ({e.count})"
        lines.append(f"  \"{e.src}\" -> \"{e.dst}\" [label=\"{label}\"];\n")
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_grouped(
    sessions: List[Tuple[str, List[Step]]],
    edges: Dict[Tuple[str, str, str], GraphEdge],
    path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Session Cross-Reference Analysis")
    lines.append("")
    for i, (question, steps) in enumerate(sessions, 1):
        lines.append(f"## Question {i}: {question}")
        for st in steps:
            if st.error:
                continue
            head = f"{st.index:02d}. {st.name} ({st.tool})"
            detail = []
            if st.collection:
                detail.append(f"collection={st.collection}")
            if st.query:
                detail.append(f"query=\"{st.query}\"")
            if st.ref_prop:
                detail.append(f"refProp={st.ref_prop}")
            if st.traversed_to:
                detail.append(f"-> {st.traversed_to}")
            lines.append(f"- {head}: " + ", ".join(detail))
            if st.traversed_items:
                lines.append(f"  - items: {', '.join(st.traversed_items)}")
        lines.append("")
    lines.append("## Cross-reference edges (all sessions)")
    for e in sorted(edges.values(), key=lambda x: (-x.count, x.src, x.prop)):
        lines.append(f"- {e.src} --{e.prop}--> {e.dst}  (x{e.count})")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Analyze agent session logs for Weaviate cross-references")
    ap.add_argument("--input", required=True, help="Path to agent_session_messages.jsonl")
    ap.add_argument("--dot", help="Write Graphviz DOT to this path")
    ap.add_argument("--markdown", help="Write Markdown report to this path")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    sessions_data: List[Tuple[str, List[Step]]] = []
    all_steps: List[Step] = []
    for obj in parse_jsonl(input_path):
        items = obj.get("session_items") or []
        question = obj.get("question")
        if not question and isinstance(items, list):
            # Fallback: first user message content
            for it in items:
                if it.get("role") == "user" and isinstance(it.get("content"), str):
                    question = it.get("content")
                    break
        if not question:
            question = "<no question>"
        if isinstance(items, list):
            steps = parse_session_items(items)
            sessions_data.append((question, steps))
            all_steps.extend(steps)

    # Build graph
    edges = build_graph(all_steps)

    # Console summary
    print("== Ordered navigation trace (per question) ==")
    for i, (question, steps) in enumerate(sessions_data, 1):
        print(f"\n-- Question {i}: {question}")
        for st in steps:
            if st.error:
                continue
            bits = [f"{st.index:02d}", st.name]
            if st.collection:
                bits.append(f"[{st.collection}]")
            if st.query:
                bits.append(f"q=\"{st.query}\"")
            if st.ref_prop:
                bits.append(f"ref={st.ref_prop}")
            if st.traversed_to:
                bits.append(f"-> {st.traversed_to}")
            print(" ".join(bits))
            if st.traversed_items:
                print("   items:", ", ".join(st.traversed_items))

    print("\n== Cross-reference edges ==")
    for e in sorted(edges.values(), key=lambda x: (-x.count, x.src, x.prop)):
        print(f"{e.src} --{e.prop}--> {e.dst}  (x{e.count})")

    if args.dot:
        write_dot(edges, Path(args.dot))
        print(f"\nDOT written to: {args.dot}")
    if args.markdown:
        write_markdown_grouped(sessions_data, edges, Path(args.markdown))
        print(f"Markdown written to: {args.markdown}")


if __name__ == "__main__":
    main()
