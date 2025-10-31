# metadata_plugin.py
import re, json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

__all__ = ["minirag_generate_metadata"]

# ---- optional validator deps (fail-gracefully) ----
try:
    from stdnum import iban as std_iban  # type: ignore
    from stdnum import bic as std_bic  # type: ignore
    from stdnum.eu import vat as std_euvat  # type: ignore
    from stdnum.pt import nif as std_ptnif  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    std_iban = std_bic = std_euvat = std_ptnif = None

try:
    import phonenumbers  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    phonenumbers = None

try:
    import dateparser  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dateparser = None
try:  # richer date searching (optional)
    from dateparser.search import search_dates  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    search_dates = None

# ---- regexes & small helpers ----
_PT_STOPWORDS = set("""
a ao aos as à às com da das de do dos e em entre para por sem sob sobre
um uma umas uns o os que é ser foi eram será serão pelo pela pelas pelos
na nas no nos num numa nuns numas este esta isto esse essa isso aquele
aquela aquilo aqui ali e/ou não sim mais menos muito muita muitos muitas
pouco pouca poucos poucas quando onde como porque porquê qual quais quem
cujo cuja cujos cujas até desde já só tão também porém contudo portanto
assim se então cada outro outra outros outras seu sua seus suas nosso
nossa nossos nossas vosso vossa vossos vossas
""".split())

# Emails / phones / codes
EMAIL_PAT   = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PO_PAT      = re.compile(r"\b(?:PO|Purchase\s*Order|Encomenda)\s*[:#]?\s*([A-Z0-9\-\/]{3,})", re.IGNORECASE)
CP_PT_PAT   = re.compile(r"\b\d{4}-\d{3}\b")  # Código Postal PT
IBAN_PAT    = re.compile(r"\b[A-Z]{2}[0-9A-Z]{13,32}\b")
BIC_PAT     = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b")
EU_VAT_PAT  = re.compile(r"\b([A-Z]{2}[A-Z0-9]{8,12})\b")  # e.g., PT123456789

# Money, invoice, dates
EUR_PAT     = re.compile(r"(?:(?:EUR|€)\s*)?([0-9]{1,3}(?:\.[0-9]{3})*(?:,[0-9]{2})|[0-9]+(?:\.[0-9]{2}))")
INV_PAT     = re.compile(r"\b(?:(?:FT|FAT|FATURA|FATURAÇÃO|INV)[\s_\-\/]*)?\d{2,4}[\/\-]\d{1,6}\b", re.IGNORECASE)
DATE_TOKEN  = re.compile(r"(\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b)")
DUE_PAT     = re.compile(r"\b(?:venc(?:imento)?|due\s*date|pagamento\s*até|payment\s*due)\s*[:\-]?\s*(.+)", re.IGNORECASE)
TERMS_PAT   = re.compile(r"\b(?:termos\s*de\s*pagamento|payment\s*terms|prazo\s*de\s*pagamento)\s*[:\-]?\s*(.+)", re.IGNORECASE)
# Addresses (Portuguese street types + CP-PT)
ADDR_PAT = re.compile(r"\b(?:Rua|Av\\.?|Avenida|Praça|Praceta|Travessa|Estrada|Rotunda|Alameda)\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\w\s,ºª\-]+", re.IGNORECASE)
POSTAL_PAT = re.compile(r"\b\d{4}-\d{3}\b")  # Código Postal PT
# Purchase order / project codes
PROJECT_PAT = re.compile(r"\b(?:Proj(?:ecto|eto)?|Project|Código\s*Proj(?:\.|eto)?|Cod\.?\s*Proj)\s*[:#]?\s*([A-Z0-9\-_/]{3,})", re.IGNORECASE)
CUSTOMER_REF_PAT = re.compile(r"\b(?:Ref(?:er[êe]ncia)?\s*(?:Cliente)?|Customer\s*Ref)\s*[:#]?\s*([A-Z0-9\-_/]{3,})", re.IGNORECASE)
# Contract numbers
CONTRACT_PAT = re.compile(r"\b(?:Contrato|Agreement|Ctr)\s*[:#]?\s*([A-Z0-9\-_/]{3,})", re.IGNORECASE)
# Dates in headers (issue date, validity)
ISSUE_DATE_PAT = re.compile(r"\b(?:Data\s*(?:Emiss[aã]o)?|Issue\s*Date)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", re.IGNORECASE)
VALIDITY_DATE_PAT = re.compile(r"\b(?:Validade|Valid\s*Until|Expira(?:tion)?)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", re.IGNORECASE)

# Doc type, sections, entity hints
DEF_TIPO_RULES = [
    ("fatura",    re.compile(r"\bfatur(a|ação)|invoice|FT\b", re.IGNORECASE)),
    ("contrato",  re.compile(r"\bcontrat(o|ual)|agreement\b", re.IGNORECASE)),
    ("memorando", re.compile(r"\bmemorando|memo\b", re.IGNORECASE)),
    ("relatorio", re.compile(r"\brelat[oó]rio|report\b", re.IGNORECASE)),
]
DEF_SECTIONS = [
    ("Resumo",    re.compile(r"\b(Resumo|Abstract)\b", re.IGNORECASE)),
    ("Cláusulas", re.compile(r"\b(Cláusulas?|Condições Gerais)\b", re.IGNORECASE)),
    ("Objetivo",  re.compile(r"\b(Objeto|Objetivo|Scope)\b", re.IGNORECASE)),
    ("Pagamento", re.compile(r"\b(Pagamento|Faturação|Preços|Payment)\b", re.IGNORECASE)),
]
ORG_SUFFIXES = re.compile(r"\b(LDA|S\.?A\.?|SA|Sociedade|Unipessoal|Limitada)\b", re.IGNORECASE)
ENTITY_HINT  = re.compile(r"^[\t ]*([A-Z0-9][A-Z0-9\-&., ]{2,60})(?:\bLDA\b|\bS\.?A\.?\b|\bUNIPESSOAL\b|\bLIMITADA\b).*?$", re.MULTILINE)

# Path → fluxo/etapa hints
H_FLUXO = [("Financeiro", re.compile(r"/fin[aâ]n?", re.IGNORECASE)),
           ("Operações",  re.compile(r"/oper[aã]o|/ops", re.IGNORECASE)),
           ("Jurídico",   re.compile(r"/jur[ií]d", re.IGNORECASE))]
H_ETAPA = [("Aprovação",  re.compile(r"aprova", re.IGNORECASE)),
           ("Revisão",    re.compile(r"revis", re.IGNORECASE)),
           ("Assinatura", re.compile(r"assina", re.IGNORECASE))]

def _top_keywords(text: str, k: int = 15) -> List[str]:
    toks = re.findall(r"[\wÀ-ÿ]{2,}", text.lower())
    toks = [t for t in toks if t not in _PT_STOPWORDS and not t.isdigit()]
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    # Sort by frequency then alphabetically for deterministic order
    return [w for w in sorted(freq.keys(), key=lambda t: (-freq[t], t))[:k]]

def _parse_date_token(s: str) -> Optional[str]:
    m = DATE_TOKEN.search(s)
    if not m: return None
    tok = m.group(1)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
        try: return datetime.strptime(tok, fmt).date().isoformat()
        except Exception: pass
    if dateparser:
        dt = dateparser.parse(tok, languages=["pt","en"])
        if dt: return dt.date().isoformat()
    return None

def _detect_tipo(text: str) -> Optional[str]:
    for label, pat in DEF_TIPO_RULES:
        if pat.search(text): return label
    return None

def _sections(text: str) -> List[Dict[str, Any]]:
    out = []
    for name, pat in DEF_SECTIONS:
        m = pat.search(text)
        if m: out.append({"name": name, "char_offset": m.start()})
    return out

def _guess_entity(text: str) -> Tuple[Optional[str], Optional[str]]:
    m = ENTITY_HINT.search(text)
    if m: return None, m.group(1).strip(" -.,")
    m2 = ORG_SUFFIXES.search(text)
    if m2:
        line = text[max(0, m2.start()-60):m2.end()+5]
        return None, line.strip()
    return None, None

def _parse_amount(text: str) -> Optional[Dict[str, Any]]:
    m = EUR_PAT.search(text)
    if not m: return None
    val = m.group(1).replace(".","").replace(",",".")
    try:  return {"value": float(val), "currency": "EUR"}
    except: return None

def _find_emails(text: str) -> List[str]:
    return sorted(set(EMAIL_PAT.findall(text)))

def _find_phones(text: str, country_default: str = "PT") -> List[str]:
    res: List[str] = []
    if phonenumbers:
        for m in re.finditer(r"[+()\d][\d\s().\-]{6,}", text):
            try:
                num = phonenumbers.parse(m.group(), country_default)
                if phonenumbers.is_valid_number(num):
                    res.append(phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164))
            except: pass
    # fallback very light
    if not res:
        res = sorted(set(re.findall(r"\+?\d[\d\s\-().]{7,}\d", text)))
    return sorted(set(res))

def _find_ibans(text: str) -> List[str]:
    cands = set(IBAN_PAT.findall(text))
    if std_iban:
        return sorted({std_iban.compact(x) for x in cands if std_iban.is_valid(x)})
    return sorted(cands)

def _find_bics(text: str) -> List[str]:
    cands = set(BIC_PAT.findall(text))  # pattern returns group; we need full match – adjust:
    # safer: rescan with finditer to capture full string
    res = set(m.group(0) for m in re.finditer(BIC_PAT, text))
    if std_bic:
        out = []
        for b in res:
            try:
                out.append(b)  # stdnum.bic has minimal validation
            except: pass
        return sorted(set(out))
    return sorted(res)

def _find_vats(text: str) -> List[str]:
    res = set()
    for m in EU_VAT_PAT.finditer(text):
        vat = m.group(1)
        # Prefer strict validation when available
        if vat.startswith("PT") and std_ptnif:
            try:
                std_ptnif.validate(vat[2:])
                res.add(vat)
                continue
            except: pass
        if std_euvat:
            try:
                std_euvat.validate(vat)
                res.add(vat)
                continue
            except: pass
        res.add(vat)  # fallback keep
    return sorted(res)

def _find_po(text: str) -> Optional[str]:
    m = PO_PAT.search(text)
    return m.group(1).strip() if m else None

def _find_due_date(text: str) -> Optional[str]:
    # capture a right-hand side token then parse; len cap to avoid giant lines
    m = DUE_PAT.search(text)
    if not m: return None
    frag = m.group(1)[:80]
    if dateparser:
        dt = dateparser.parse(frag, languages=["pt","en"])
        if dt: return dt.date().isoformat()
    # fallback token parse
    return _parse_date_token(frag)

def _find_all_dates(text: str) -> List[str]:
    """Extract date expressions and normalize to ISO (YYYY-MM-DD) when possible.

    Strategy (ordered, unique):
    1. If dateparser.search.search_dates is available, use it to locate a broad set of date
       expressions (supports Portuguese & English). For each hit, keep only the date part
       (drop time) and append in order of first appearance.
    2. Fallback / supplement: scan with DATE_TOKEN regex and parse each token via _parse_date_token.
    3. Return list preserving the first appearance order (no sorting to avoid reordering semantic flow).
    """
    seen: set[str] = set()
    ordered: List[str] = []

    # 1) Broad search (if available)
    if search_dates:
        try:
            results = search_dates(
                text,
                languages=["pt", "en"],
                settings={
                    "RETURN_AS_TIMEZONE_AWARE": False,
                    # Avoid false positives like standalone numbers by requiring day/month keywords implicitly
                    # (dateparser doesn't expose a simple strict mode; rely on default heuristics)
                },
            ) or []
            # search_dates returns list[tuple[str, datetime]] in textual order already
            for match_txt, dt in results:
                # Normalize to date only
                if not dt:  # defensive
                    continue
                iso = dt.date().isoformat()
                if iso not in seen:
                    seen.add(iso)
                    ordered.append(iso)
        except Exception:
            pass  # graceful fallback to regex below

    # 2) Regex tokens (acts as fallback or to catch things search_dates missed due to pattern boundary)
    for m in DATE_TOKEN.finditer(text):
        raw = m.group(1)
        norm = _parse_date_token(raw)
        if norm and norm not in seen:
            seen.add(norm)
            ordered.append(norm)

    return ordered

def _find_payment_terms(text: str) -> Optional[str]:
    m = TERMS_PAT.search(text)
    if m: return m.group(1).strip()[:120]
    # fallback: “net 30/45/60”
    m2 = re.search(r"\bnet\s*(\d{1,3})\b", text, re.IGNORECASE)
    if m2: return f"net {m2.group(1)}"
    return None

def _find_addresses(text: str) -> List[str]:
    return sorted(set(m.group(0).strip() for m in ADDR_PAT.finditer(text)))

def _find_postal_codes(text: str) -> List[str]:
    return sorted(set(m.group(0).strip() for m in POSTAL_PAT.finditer(text)))

def _find_project_codes(text: str) -> List[str]:
    return sorted(set(m.group(1).strip() for m in PROJECT_PAT.finditer(text)))

def _find_customer_refs(text: str) -> List[str]:
    return sorted(set(m.group(1).strip() for m in CUSTOMER_REF_PAT.finditer(text)))

def _find_contract_numbers(text: str) -> List[str]:
    return sorted(set(m.group(1).strip() for m in CONTRACT_PAT.finditer(text)))

def _find_issue_date(text: str) -> Optional[str]:
    m = ISSUE_DATE_PAT.search(text)
    return _parse_date_token(m.group(1)) if m else None

def _find_validity_date(text: str) -> Optional[str]:
    m = VALIDITY_DATE_PAT.search(text)
    return _parse_date_token(m.group(1)) if m else None



def _fluxo_from_path(path: str) -> Tuple[Optional[str], Optional[str]]:
    for name, pat in H_FLUXO:
        if pat.search(path):
            return None, name
    return None, None

def _etapa_from_path(path: str) -> Tuple[Optional[str], Optional[str]]:
    for name, pat in H_ETAPA:
        if pat.search(path):
            return None, name
    return None, None

# ---- public function ----
def minirag_generate_metadata(doc_id: str, text: str, path: Any) -> Dict[str, Any]:
    # best-effort title
    title = next((ln.strip()[:120] for ln in text.splitlines() if ln.strip()), None)

    tipo = _detect_tipo(text)
    inv  = (m.group(0).strip() if (m := INV_PAT.search(text)) else None)
    created_at = _parse_date_token(text)

    amount = _parse_amount(text)
    emails = _find_emails(text)
    phones = _find_phones(text, country_default="PT")
    ibans  = _find_ibans(text)
    bics   = _find_bics(text)
    vats   = _find_vats(text)
    po_no  = _find_po(text)
    due    = _find_due_date(text)
    terms  = _find_payment_terms(text)
    addresses = _find_addresses(text)
    project_codes = _find_project_codes(text)
    customer_refs = _find_customer_refs(text)
    contract_numbers = _find_contract_numbers(text)
    issue_date = _find_issue_date(text)
    validity_date = _find_validity_date(text)
    postal_codes = _find_postal_codes(text)
    all_dates = _find_all_dates(text)

    entidade_id, entidade_name = _guess_entity(text)
    # Normalize path once (accept Path or str); use forward slashes for pattern matching
    path_str = str(path)
    path_for_match = path_str.replace("\\", "/")
    fluxo_id, fluxo_name = _fluxo_from_path(path_for_match)
    etapa_id, etapa_name = _etapa_from_path(path_for_match)

    meta = {
        "doc_id": doc_id,
        "core": {
            "title": title, "lang": None, "mime": "application/pdf",
            "created_at": created_at, "source_path": path_str,
            "pages": None, "text_quality": {}
        },
        "business": {
            "tipo_doc": tipo,
            "entidade_id": entidade_id, "entidade_name": entidade_name,
            "fluxo_id": fluxo_id, "fluxo_name": fluxo_name,
            "etapa_id": etapa_id, "etapa_name": etapa_name,
            "periodo_ref": None, "n_fatura": inv, "montante": amount,
            # Keep both a PT NIF and a generic VATs list
            "nif": next((v[2:] for v in vats if v.startswith("PT")), None)
        },
        "content_signals": {
            "keywords": _top_keywords(text, 15),
            "sections": _sections(text),
            "entities": ([{"type": "ORG", "text": entidade_name}] if entidade_name else [])
        },
        "references": {
            "project_codes": project_codes,
            "customer_refs": customer_refs,
            "contract_numbers": contract_numbers
        },
        "address": {
            "list": addresses,
            "postal_codes": postal_codes
        },
        "dates": {
        "issue_date": issue_date,
        "validity_date": validity_date,
        "all_dates": all_dates
        },
        "payment": {
            "po_number": po_no,
            "due_date": due,
            "terms": terms,
            "iban_list": ibans,
            "bic_list": bics
        },
        "contacts": {
            "emails": emails,
            "phones": phones
        },
        "identifiers": {
            "vat_ids": vats,
            "postal_codes_pt": sorted(set(CP_PT_PAT.findall(text)))
        },
        "graph": {
            "entidade_id": entidade_id, "fluxo_id": fluxo_id, "etapa_id": etapa_id,
            "parent_pasta_id": None, "relates_to": []
        },
        "security": {
            "acl": ["ROLE:default"],
            "pii": (["EMAIL"] if emails else []) + (["PHONE"] if phones else []) + (["IBAN"] if ibans else []),
            "retention_class": None
        },
        "provenance": {
            "ingested_by": "minirag-metadata-plugin@0.2",
            "extractors": [
                {"name": "rules.tipo_doc", "confidence": 0.7 if tipo else 0.0},
                {"name": "rules.invoice",  "confidence": 0.6 if inv else 0.0},
                {"name": "rules.amount",   "confidence": 0.8 if amount else 0.0},
                {"name": "rules.vat",      "confidence": 0.9 if vats else 0.0},
                {"name": "rules.iban",     "confidence": 0.9 if ibans else 0.0},
                {"name": "rules.bic",      "confidence": 0.7 if bics else 0.0},
                {"name": "rules.email",    "confidence": 0.95 if emails else 0.0},
                {"name": "rules.phone",    "confidence": 0.9 if phones else 0.0},
                {"name": "rules.po",       "confidence": 0.6 if po_no else 0.0},
                {"name": "rules.due",      "confidence": 0.6 if due else 0.0},
            ]
        }
    }
    # JSON-serializable normalization
    return json.loads(json.dumps(meta, ensure_ascii=False))
