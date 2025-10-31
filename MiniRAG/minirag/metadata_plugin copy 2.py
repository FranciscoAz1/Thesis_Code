# metadata_plugin.py
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional dependencies
try:
    from stdnum import iban as std_iban  # type: ignore
    from stdnum.pt import nif as std_ptnif  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    std_iban = std_ptnif = None

try:
    import phonenumbers  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    phonenumbers = None

try:  # pragma: no cover - optional dependency
    from dateparser.search import search_dates  # type: ignore
    import dateparser  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    dateparser = None
    search_dates = None

# Required dependency: presidio_analyzer
from presidio_analyzer import AnalyzerEngine  # type: ignore
from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult  # type: ignore
from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore
_HAS_PRESIDIO = True

__all__ = ["minirag_generate_metadata"]

# Constants
_PRESIDIO_ANALYZER = None  # lazy singleton

# Module-level logger
logger = logging.getLogger("minirag.metadata")

# Regex patterns (only used ones)
DATE_TOKEN = re.compile(r"(\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b)")
DUE_PAT = re.compile(r"\b(?:venc(?:imento)?|due\s*date|pagamento\s*até|payment\s*due)\s*[:\-]?\s*(.+)", re.IGNORECASE)

# Path hints
H_FLUXO = [
    ("Financeiro", re.compile(r"/fin[aâ]n?", re.IGNORECASE)),
    ("Operações", re.compile(r"/oper[aã]o|/ops", re.IGNORECASE)),
    ("Jurídico", re.compile(r"/jur[ií]d", re.IGNORECASE)),
]
H_ETAPA = [
    ("Aprovação", re.compile(r"aprova", re.IGNORECASE)),
    ("Revisão", re.compile(r"revis", re.IGNORECASE)),
    ("Assinatura", re.compile(r"assina", re.IGNORECASE)),
]


def _get_presidio_analyzer():  # pragma: no cover - runtime optional
    """Instantiate a Presidio AnalyzerEngine lazily.

    We avoid mandatory heavy NLP model downloads. If spaCy models are missing,
    Presidio will attempt to download / fail; we catch and degrade gracefully.
    """
    global _PRESIDIO_ANALYZER
    if _PRESIDIO_ANALYZER:
        return _PRESIDIO_ANALYZER
    
    # Suppress verbose Presidio logging
    import logging
    presidio_logger = logging.getLogger("presidio-analyzer")
    original_level = presidio_logger.level
    presidio_logger.setLevel(logging.WARNING)  # Suppress INFO messages
    
    try:
        # Attempt to build a Portuguese-only spaCy engine (pt_core_news_sm must be installed externally).
        engine = None
        if NlpEngineProvider is not None:
            try:
                logger.info("Attempting to load spaCy model 'pt_core_news_sm' for Presidio analyzer")
                cfg = {
                    "nlp_engine_name": "spacy", 
                    "models": [{"lang_code": "pt", "model_name": "pt_core_news_sm"}],
                    "ner_model_configuration": {
                        "labels_to_ignore": ["MISC"]
                    }
                }
                engine = NlpEngineProvider(nlp_configuration=cfg).create_engine()
                logger.info("Loaded Portuguese spaCy model 'pt_core_news_sm' successfully")
            except Exception as e:
                engine = None
                logger.warning("Could not load spaCy model 'pt_core_news_sm'; falling back. Error: %s", e)
        if engine is not None:
            _PRESIDIO_ANALYZER = AnalyzerEngine(nlp_engine=engine, supported_languages=["pt"])  # type: ignore
        else:
            # Fallback create analyzer (may still load EN internally) but restrict supported_languages when possible.
            try:
                logger.info("Creating Presidio AnalyzerEngine fallback (no dedicated Portuguese spaCy model)")
                # Try to create with default configuration that ignores MISC
                if NlpEngineProvider is not None:
                    try:
                        cfg_fallback = {
                            "nlp_engine_name": "spacy", 
                            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                            "ner_model_configuration": {
                                "labels_to_ignore": ["MISC"]
                            }
                        }
                        engine = NlpEngineProvider(nlp_configuration=cfg_fallback).create_engine()
                        _PRESIDIO_ANALYZER = AnalyzerEngine(nlp_engine=engine, supported_languages=["pt"])
                    except Exception:
                        _PRESIDIO_ANALYZER = AnalyzerEngine(supported_languages=["pt"])
                else:
                    _PRESIDIO_ANALYZER = AnalyzerEngine(supported_languages=["pt"])
            except Exception:
                logger.info("Creating fully default Presidio AnalyzerEngine (final fallback)")
                _PRESIDIO_ANALYZER = AnalyzerEngine()
        _presidio_register_custom(_PRESIDIO_ANALYZER)
        if _PRESIDIO_ANALYZER:
            logger.debug("Presidio analyzer initialized with custom recognizers: %s", [r.name for r in _PRESIDIO_ANALYZER.registry.recognizers][-5:])
            # Log which built-in recognizers are loaded
            builtin_recognizers = [r.name for r in _PRESIDIO_ANALYZER.registry.recognizers if not r.name.endswith('Recognizer')]
            if builtin_recognizers:
                logger.info("Built-in Presidio recognizers loaded: %s", builtin_recognizers[:10])  # Show first 10
    except Exception as e:
        logger.error("Failed to initialize Presidio analyzer: %s", e)
        _PRESIDIO_ANALYZER = None
    finally:
        # Keep the logging level suppressed to prevent verbose Presidio messages during analysis
        presidio_logger.setLevel(logging.WARNING)
    
    return _PRESIDIO_ANALYZER

def _presidio_register_custom(analyzer):  # pragma: no cover - optional
    """Register custom recognizers for domain-specific patterns (idempotent).

    Adds Portuguese (pt) contextual keywords to improve disambiguation/scoring.
    """
    existing = {r.name for r in analyzer.registry.recognizers}
    def add(name: str, entity: str, regex: str, score: float = 0.6, context: Optional[List[str]] = None):
        if name in existing:
            return
        try:
            pat = Pattern(name=f"{name}_pat", regex=regex, score=score)
            recog = PatternRecognizer(name=name, supported_entity=entity, patterns=[pat], context=context or [])
            analyzer.registry.add_recognizer(recog)
        except Exception:
            return
    # Use explicit raw regex strings (simplifies reading without referencing compiled objects)
    add("IBANRecognizer", "IBAN", r"\b[A-Z]{2}[0-9A-Z]{13,32}\b", context=["iban","nib","conta","banco","bank"])
    add("BICRecognizer", "BIC", r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b", context=["bic","swift","bank","banco"])
    add("VATRecognizer", "VAT_ID", r"\b([A-Z]{2}[A-Z0-9]{8,12})\b", context=["vat","nif","contribuinte","iva","tax","fiscal"])
    add("PORecognizer", "PO_NUMBER", r"\b(?:PO|Purchase\s*Order|Encomenda)\s*[:#]?\s*([A-Z0-9\-\/]{3,})", context=["po","encomenda","pedido","purchase","order"])
    add("InvoiceRecognizer", "INVOICE_NO", r"\b(?:(?:FT|FAT|FATURA|FATURAÇÃO|INV)[\s_\-\/]*)?\d{2,4}[\/\-]\d{1,6}\b", context=["fatura","factura","invoice","nota","ft","faturação"])
    add("AmountEURRecognizer", "AMOUNT_EUR", r"([0-9]{1,3}(?:\.[0-9]{3})*(?:,[0-9]{2})|[0-9]+(?:\.[0-9]{2}))", context=["total","montante","valor","eur","euro","€","liquido","bruto"])
    add("DueDateRecognizer", "DUE_DATE", r"\b(?:venc(?:imento)?|due\s*date|pagamento\s*até|payment\s*due)\s*[:\-]?\s*(.+)", context=["vencimento","due","pagamento","até","limite"])
    add("PaymentTermsRecognizer", "PAYMENT_TERMS", r"\b(?:termos\s*de\s*pagamento|payment\s*terms|prazo\s*de\s*pagamento)\s*[:\-]?\s*(.+)", context=["termos","pagamento","prazo","condições"])
    add("AddressPTRecognizer", "ADDRESS_PT", r"\b(?:Rua|Av\\\.?|Avenida|Praça|Praceta|Travessa|Estrada|Rotunda|Alameda)\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\w\s,ºª\-]+", context=["endereço","morada","local","rua","avenida"])
    add("ProjectCodeRecognizer", "PROJECT_CODE", r"\b(?:Proj(?:ecto|eto)?|Project|Código\s*Proj(?:\. |eto)?|Cod\.\?\s*Proj)\s*[:#]?\s*([A-Z0-9\-_/]{3,})", context=["projeto","projecto","project","código","ref","referência"])
    add("CustomerRefRecognizer", "CUSTOMER_REF", r"\b(?:Ref(?:er[êe]ncia)?\s*(?:Cliente)?|Customer\s*Ref)\s*[:#]?\s*([A-Z0-9\-_/]{3,})", context=["cliente","customer","ref","referência"])
    add("ContractNumberRecognizer", "CONTRACT_NUMBER", r"\b(?:Contrato|Agreement|Ctr)\s*[:#]?\s*([A-Z0-9\-_/]{3,})", context=["contrato","agreement","ctr"])
    add("IssueDateRecognizer", "ISSUE_DATE", r"\b(?:Data\s*(?:Emiss[aã]o)?|Issue\s*Date)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", context=["emissão","issue","data"])
    add("ValidityDateRecognizer", "VALIDITY_DATE", r"\b(?:Validade|Valid\s*Until|Expira(?:tion)?)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", context=["validade","expira","válido","expiry","until"])
    add("PostalCodePTRecognizer", "POSTAL_CODE_PT", r"\b\d{4}-\d{3}\b", context=["código","postal","cp"])
    add("SectionResumoRecognizer", "SECTION_RESUMO", r"\b(Resumo|Abstract)\b", context=["resumo","abstract","summary"])
    add("SectionClausulasRecognizer", "SECTION_CLAUSULAS", r"\b(Cláusulas?|Condições Gerais)\b", context=["cláusulas","condições","gerais","clauses","general","conditions"])
    add("SectionObjetivoRecognizer", "SECTION_OBJETIVO", r"\b(Objeto|Objetivo|Scope)\b", context=["objeto","objetivo","scope","purpose"])
    add("SectionPagamentoRecognizer", "SECTION_PAGAMENTO", r"\b(Pagamento|Faturação|Preços|Payment)\b", context=["pagamento","faturação","preços","payment","billing","pricing"])
    add("DocTypeFaturaRecognizer", "DOC_TYPE_FATURA", r"\b(?:fatur(a|ação)|invoice|FT)\b", context=["fatura","factura","invoice","nota","ft","faturação"])
    add("DocTypeContratoRecognizer", "DOC_TYPE_CONTRATO", r"\b(?:contrat(o|ual)|agreement)\b", context=["contrato","agreement","ctr"])
    add("DocTypeMemorandoRecognizer", "DOC_TYPE_MEMORANDO", r"\b(?:memorando|memo)\b", context=["memorando","memo"])
    add("DocTypeRelatorioRecognizer", "DOC_TYPE_RELATORIO", r"\b(?:relat[oó]rio|report)\b", context=["relatório","relatorio","report"])
    add("OrgEntityRecognizer", "ORG_ENTITY", r"^[\t ]*([A-Z0-9][A-Z0-9\-&., ]{2,60})(?:\bLDA\b|\bS\.?A\.?\b|\bUNIPESSOAL\b|\bLIMITADA\b).*?$", context=["empresa","sociedade","lda","sa","limitada","unipessoal"])
    add("OrgSuffixRecognizer", "ORG_SUFFIX", r"\b(LDA|S\.?A\.?|SA|Sociedade|Unipessoal|Limitada)\b", context=["empresa","sociedade","lda","sa","limitada","unipessoal"])
    # Portuguese honorifics and titles to improve PERSON detection
    add("PortugueseTitlesRecognizer", "PERSON_TITLE", r"\b(?:Sr\.?|Sra\.?|Dr\.?|Dra\.?|Eng\.?|Prof\.?|Exmo\.?|Ilmo\.?|Ilma\.?|Arq\.?|Adv\.?|Cel\.?|Gen\.?|Maj\.?|Cap\.?|Ten\.?|Sgt\.?)\b", 
        context=["nome","assinatura","responsável","contacto","contato","pessoa","indivíduo","cidadão"])
    # Name prefixes that often appear before actual names
    add("PortugueseNamePrefixesRecognizer", "PERSON_PREFIX", r"\b(?:Nome|Assinatura|Responsável|Contacto|Contato|Destinatário|Remetente|Representante|Autor|Proprietário|Cliente)\s*[:\-]?\s*[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*", 
        context=["documento","contrato","fatura","carta","comunicado"])
    # Portuguese name patterns (common first + last name combinations)
    add("PortugueseNamePatternRecognizer", "PERSON_NAME", r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\b", 
        context=["assinatura","nome","pessoa","cliente","fornecedor","empresa","contrato"])
    # Portuguese full names with middle names (e.g., "João Carlos Silva", "Maria José Santos")
    add("PortugueseFullNameRecognizer", "PERSON_FULL_NAME", r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\b", 
        context=["assinatura","nome","pessoa","cliente","fornecedor","empresa","contrato","responsável","contacto"])
    # Portuguese names with titles (e.g., "Sr. João Silva", "Dra. Maria Santos")
    add("PortugueseTitledNameRecognizer", "PERSON_TITLED", r"\b(?:Sr\.?|Sra\.?|Dr\.?|Dra\.?|Eng\.?|Prof\.?|Exmo\.?|Ilmo\.?|Ilma\.?|Arq\.?|Adv\.?)\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*\b", 
        context=["assinatura","nome","pessoa","cliente","fornecedor","empresa","contrato","responsável","contacto"])
    # Portuguese names with academic titles (e.g., "Prof. Dr. João Silva", "Eng.ª Maria Santos") 
    add("PortugueseAcademicNameRecognizer", "PERSON_ACADEMIC", r"\b(?:Prof\.?\s+)?(?:Dr\.?|Dra\.?|Eng\.?|Eng\.ª|Mestre|Mestra|Lic\.?|PhD)\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*\b", 
        context=["docente","professor","doutor","engenheiro","mestre","licenciado","universidade","academia"])
    # Portuguese military/police titles with names
    add("PortugueseMilitaryNameRecognizer", "PERSON_MILITARY", r"\b(?:Cel\.?|Gen\.?|Maj\.?|Cap\.?|Ten\.?|Sgt\.?|Sarg\.?|Cabo|Sold\.?|Comandante|Major|Coronel|General)\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*\b", 
        context=["militar","policia","segurança","defesa","exército","guarda","forças","armadas"])
    # Portuguese organization patterns with legal suffixes
    add("PortugueseOrgWithSuffixRecognizer", "ORG_ENTITY_SUFFIX", r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇa-záéíóúâêôãõç\s&.,]{2,60}(?:\bLDA\b|\bS\.?A\.?\b|\bUNIPESSOAL\b|\bLIMITADA\b|\bS\.?A\.?\b|\bSociedade\b|\bUnipessoal\b|\bLimitada\b)\b", 
        context=["empresa","sociedade","lda","sa","limitada","unipessoal","firma","companhia"])
    # Portuguese document signatory patterns
    add("PortugueseSignatoryRecognizer", "PERSON_SIGNATORY", r"\b(?:Assinatura|Assinado\s*por|Por|Em\s*nome\s*de)\s*[:\-]?\s*(?:Sr\.?|Sra\.?|Dr\.?|Dra\.?|Eng\.?|Prof\.?)?\s*[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*\b", 
        context=["assinatura","assinado","documento","contrato","autorização","representante"])
    # Portuguese responsible person patterns
    add("PortugueseResponsibleRecognizer", "PERSON_RESPONSIBLE", r"\b(?:Responsável|Representante|Contacto|Contato|Gestor|Director|Diretor|Coordenador)\s*[:\-]?\s*(?:Sr\.?|Sra\.?|Dr\.?|Dra\.?|Eng\.?)?\s*[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*\b", 
        context=["responsável","representante","contacto","contato","gestor","director","coordenador"])
    # Portuguese client/supplier name patterns
    add("PortugueseClientSupplierRecognizer", "PERSON_CLIENT_SUPPLIER", r"\b(?:Cliente|Fornecedor|Prestador|Contratante|Contratado)\s*[:\-]?\s*(?:Sr\.?|Sra\.?|Dr\.?|Dra\.?|Eng\.?)?\s*[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*\b", 
        context=["cliente","fornecedor","prestador","contratante","contratado","empresa","pessoa"])
    # Portuguese company representative patterns
    add("PortugueseCompanyRepRecognizer", "PERSON_COMPANY_REP", r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇa-záéíóúâêôãõç\s&.,]{2,60}(?:\bLDA\b|\bS\.?A\.?\b|\bUNIPESSOAL\b|\bLIMITADA\b)\s*(?:por|representad[ao]\s*por|através\s*de)\s*(?:Sr\.?|Sra\.?|Dr\.?|Dra\.?|Eng\.?)?\s*[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*\b", 
        context=["representada","representado","através","empresa","sociedade","lda","sa"])

    # Date parser recognizer (dynamic) using dateparser.search_dates for Portuguese
    if dateparser and search_dates and 'DateParserRecognizer' not in existing and EntityRecognizer and RecognizerResult:
        class DateParserRecognizer(EntityRecognizer):  # type: ignore
            def __init__(self):  # pragma: no cover - simple init
                super().__init__(supported_entities=["DATE_GENERIC"], name="DateParserRecognizer")
                # Ensure Presidio runs this for Portuguese language calls
                self.supported_language = "pt"
            def load(self):  # pragma: no cover
                return
            def analyze(self, text, entities, nlp_artifacts=None):  # pragma: no cover - runtime only
                if entities and "DATE_GENERIC" not in entities:
                    return []
                out = []
                results = []
                if search_dates:
                    try:
                        results = search_dates(text, languages=["pt","en"], settings={"RETURN_AS_TIMEZONE_AWARE": False}) or []
                    except Exception:
                        results = []
                seen_spans = set()
                for match_txt, dt in results[:200]:  # cap
                    if not dt:
                        continue
                    start = text.find(match_txt)
                    if start == -1:
                        continue
                    end = start + len(match_txt)
                    key = (start,end)
                    if key in seen_spans:
                        continue
                    seen_spans.add(key)
                    # heuristic score: higher if year present
                    score = 0.90 if re.search(r"\b\d{4}\b", match_txt) else 0.86
                    out.append(RecognizerResult(entity_type="DATE_GENERIC", start=start, end=end, score=score))
                return out
        try:
            analyzer.registry.add_recognizer(DateParserRecognizer())
        except Exception:
            raise ValueError("Failed to add DateParserRecognizer")
    # Find entities, names, company's affiliations
    

def _extract_with_presidio(text: str) -> Dict[str, Any]:  # pragma: no cover - optional
    """Extract PII and business entities from text using Presidio analyzer.

    Args:
        text: The text to analyze.

    Returns:
        A dictionary with extracted entities like amounts, emails, IBANs, etc.
    """
    analyzer = _get_presidio_analyzer()
    if not analyzer:
        return {}
    try:
        raw_results = analyzer.analyze(text=text, entities=None, language="pt")
        # Filter results to improve accuracy
        filtered_results = []
        for r in raw_results:
            # Skip PERSON detections with very low scores or very short matches
            if r.entity_type in ["PERSON", "PERSON_TITLED", "PERSON_FULL_NAME", "PERSON_ACADEMIC", "PERSON_MILITARY", "PERSON_ECCLESIASTICAL"]:
                snippet = text[r.start:r.end] if r.start < len(text) and r.end <= len(text) else ""
                
                # Check if this PERSON detection is near Portuguese titles/prefixes
                has_portuguese_context = False
                context_window = text[max(0, r.start-50):min(len(text), r.end+50)]
                portuguese_indicators = ["Sr.", "Sra.", "Dr.", "Dra.", "Eng.", "Prof.", "Nome:", "Assinatura:", "Responsável:", "Cliente:", "Fornecedor:", "Empresa:"]
                if any(indicator in context_window for indicator in portuguese_indicators):
                    has_portuguese_context = True
                
                # Be more lenient with PERSON detection if Portuguese context is found
                if has_portuguese_context:
                    if getattr(r, 'score', 0.0) < 0.5:  # Lower threshold with context
                        logger.debug(f"Filtered out {r.entity_type} with Portuguese context but low score: '{snippet}' with score {getattr(r, 'score', 0.0)}")
                        continue
                else:
                    if getattr(r, 'score', 0.0) < 0.7:  # Higher threshold without context
                        logger.debug(f"Filtered out low-score {r.entity_type}: '{snippet}' with score {getattr(r, 'score', 0.0)}")
                        continue
                
                if len(snippet.strip()) <= 2:
                    logger.debug(f"Filtered out short {r.entity_type}: '{snippet}'")
                    continue
            filtered_results.append(r)
        
        # Optimized deduplication: group by text content and keep highest score
        text_to_results = {}
        for r in filtered_results:
            text_content = text[r.start:r.end].strip()
            if text_content not in text_to_results:
                text_to_results[text_content] = r
            else:
                # Keep the result with higher score
                existing_score = getattr(text_to_results[text_content], 'score', 0.0)
                current_score = getattr(r, 'score', 0.0)
                if current_score > existing_score:
                    text_to_results[text_content] = r
        
        # Convert back to list
        results = list(text_to_results.values())
        logger.debug(f"Presidio analysis: {len(results)} entities detected after filtering and deduplication (from {len(raw_results)} raw, {len(filtered_results)} filtered)")
    except Exception:
        return {}
    by_type: Dict[str, List[Any]] = {}
    for r in results:
        by_type.setdefault(r.entity_type, []).append(r)

    def spans(entity: str) -> List[str]:
        return [text[r.start:r.end] for r in by_type.get(entity, []) if r.start < len(text) and r.end <= len(text)]

    # Scalar helpers (take first span)
    def first(entity: str) -> Optional[str]:
        s = spans(entity)
        return s[0].strip() if s else None

    # Amount extraction: presidio already returns just the numeric part
    amount_raw = first("AMOUNT_EUR")
    amount_val = None
    if amount_raw:
        try:
            num = amount_raw.replace(".", "").replace(",", ".")
            amount_val = {"value": float(num), "currency": "EUR"}
        except Exception:
            pass

    # IBAN validation using stdnum if present
    ibans_list = []
    if std_iban:
        for cand in spans("IBAN"):
            cand_clean = cand.strip()
            try:
                if std_iban.is_valid(cand_clean):
                    ibans_list.append(std_iban.compact(cand_clean))
                else:
                    ibans_list.append(cand_clean)
            except Exception:
                ibans_list.append(cand_clean)
    else:
        ibans_list = [cand.strip() for cand in spans("IBAN")]
    ibans_list = sorted(set(ibans_list))

    # BIC list (minimal validation)
    bics_list = sorted({c.strip() for c in spans("BIC")})

    # VAT IDs (validate PT NIF when possible)
    vat_list = []
    for cand in spans("VAT_ID"):
        vat = cand.strip()
        if vat.startswith("PT") and std_ptnif:
            try:
                std_ptnif.validate(vat[2:])
                vat_list.append(vat)
            except Exception:
                pass
        else:
            vat_list.append(vat)
    vat_list = sorted(set(vat_list))

    # Build unified PII findings (single analyze() call reused by caller)
    pii_findings = [
        {
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": round(getattr(r, "score", 0.0), 4),
            "text": text[r.start:r.end][:120] if r.start < len(text) and r.end <= len(text) else "",
            "context": text[max(0, r.start-5):min(len(text), r.end+5)].strip()
        }
        for r in results
    ]

    # Extract sections from presidio results
    sections_list = []
    section_mapping = {
        "SECTION_RESUMO": "Resumo",
        "SECTION_CLAUSULAS": "Cláusulas", 
        "SECTION_OBJETIVO": "Objetivo",
        "SECTION_PAGAMENTO": "Pagamento"
    }
    for r in results:
        if r.entity_type in section_mapping:
            sections_list.append({
                "name": section_mapping[r.entity_type],
                "char_offset": r.start
            })
    sections_list = sorted(sections_list, key=lambda x: x["char_offset"])

    # Extract document type from presidio results
    doc_type_mapping = {
        "DOC_TYPE_FATURA": "fatura",
        "DOC_TYPE_CONTRATO": "contrato",
        "DOC_TYPE_MEMORANDO": "memorando",
        "DOC_TYPE_RELATORIO": "relatorio"
    }

    return {
        "amount": amount_val,
        "emails": sorted({e for e in spans("EMAIL_ADDRESS")}),
        "phones": sorted({p for p in spans("PHONE_NUMBER")}),
        "ibans": ibans_list,
        "bics": bics_list,
        "vats": vat_list,
        "po_no": first("PO_NUMBER"),
        "invoice_no": first("INVOICE_NO"),
        "due": (lambda d: _find_due_date(d) if d else None)(first("DUE_DATE")),
        "terms": first("PAYMENT_TERMS"),
        "addresses": sorted({a.strip() for a in spans("ADDRESS_PT")}),
        "project_codes": sorted({c.strip() for c in spans("PROJECT_CODE")}),
        "customer_refs": sorted({c.strip() for c in spans("CUSTOMER_REF")}),
        "contract_numbers": sorted({c.strip() for c in spans("CONTRACT_NUMBER")}),
        "postal_codes": sorted({c.strip() for c in spans("POSTAL_CODE_PT")}),
        "doc_title": first("DOC_TITLE"),
        "sections": sections_list,
        "pii_findings": pii_findings,
    }

def _parse_date_token(s: str) -> Optional[str]:
    if not s:
        return None
    m = DATE_TOKEN.search(s)
    tok = m.group(1) if m else s.strip()[:40]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(tok, fmt).date().isoformat()
        except Exception:
            pass
    if dateparser:
        try:
            dt = dateparser.parse(tok, languages=["pt","en"])
            if not dt and tok != s.strip():  # try full original string if token subset failed
                dt = dateparser.parse(s, languages=["pt","en"])
            if dt:
                return dt.date().isoformat()
        except Exception:
            return None
    return None

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

def clean_text(s: str) -> str:
    """Lightweight normalization and cleaning for text extraction.

    Operations:
        - Normalize Windows/Mac newlines to '\n'
        - Remove zero-width/control chars (except newline/tab)
        - Collapse repeated spaces/tabs → single space
        - Collapse 3+ blank lines → double newline
        - Trim trailing spaces per line
        - Replace newlines with spaces

    Args:
        s: The input text string.

    Returns:
        The cleaned text string.
    """
    if not s:
        return ""
    # Normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Remove NULL and other control chars except \n and \t
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)
    # Collapse horizontal whitespace
    s = re.sub(r"[ \t]+", " ", s)
    # Trim spaces around newlines
    s = re.sub(r" *\n *", "\n", s)
    # Collapse many blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    # replace \n with <br>
    s = s.replace("\n", " ")
    return s

# ---- public function ----
def minirag_generate_metadata(doc_id: str, text: str, path: Any) -> Dict[str, Any]:
    """Generate metadata for a document using various extraction techniques.

    Args:
        doc_id: Unique identifier for the document.
        text: The text content of the document.
        path: The file path or path-like object of the document.

    Returns:
        A dictionary containing extracted metadata including business info,
        content signals, references, addresses, dates, payment details, etc.
    """
    # Basic doc-level signals (still lightweight even when relying on Presidio for PII)
    # clean text for better extraction
    text = clean_text(text)

    pres_ex = _extract_with_presidio(text)  # rely solely on Presidio (no regex fallback)
    amount = pres_ex.get("amount")
    emails = pres_ex.get("emails", [])
    phones = pres_ex.get("phones", [])
    ibans = pres_ex.get("ibans", [])
    bics = pres_ex.get("bics", [])
    vats = pres_ex.get("vats", [])
    po_no = pres_ex.get("po_no")
    due = pres_ex.get("due")
    terms = pres_ex.get("terms")
    addresses = pres_ex.get("addresses", [])
    project_codes = pres_ex.get("project_codes", [])
    customer_refs = pres_ex.get("customer_refs", [])
    contract_numbers = pres_ex.get("contract_numbers", [])
    postal_codes = pres_ex.get("postal_codes", [])
    doc_title_ex = pres_ex.get("doc_title")
    sections = pres_ex.get("sections", [])
    # Normalize path once (accept Path or str); use forward slashes for pattern matching
    path_str = str(path)
    path_for_match = path_str.replace("\\", "/")
    fluxo_id, fluxo_name = _fluxo_from_path(path_for_match)
    etapa_id, etapa_name = _etapa_from_path(path_for_match)

    # Reuse findings from earlier Presidio extraction (single analyzer call)
    presidio_findings = pres_ex.get("pii_findings", [])
    presidio_entity_types = sorted({f.get("entity_type") for f in presidio_findings}) if presidio_findings else []

    # Compute keywords and sections
    # postal_codes_pt is now handled by presidio

    # Build flat meta dictionary based on detected data
    meta = {}

    # Always include doc_id
    meta["doc_id"] = doc_id

    # Core fields
    if doc_title_ex:
        meta["title"] = doc_title_ex
    meta["mime"] = "application/pdf"
    if path_str:
        meta["source_path"] = path_str

    # Business fields
    if fluxo_name:
        meta["fluxo_name"] = fluxo_name
    if etapa_name:
        meta["etapa_name"] = etapa_name
    if amount:
        meta["montante"] = amount
    nif = next((v[2:] for v in vats if v.startswith("PT")), None)
    if nif:
        meta["nif"] = nif

    # Content signals
    if sections:
        meta["sections"] = sections

    # References
    if project_codes:
        meta["project_codes"] = project_codes
    if customer_refs:
        meta["customer_refs"] = customer_refs
    if contract_numbers:
        meta["contract_numbers"] = contract_numbers

    # Address
    if addresses:
        meta["addresses"] = addresses
    if postal_codes:
        meta["postal_codes"] = postal_codes

    # Payment
    if po_no:
        meta["po_number"] = po_no
    if due:
        meta["due_date"] = due
    if terms:
        meta["payment_terms"] = terms
    if ibans:
        meta["iban_list"] = ibans
    if bics:
        meta["bic_list"] = bics

    # Contacts
    if emails:
        meta["emails"] = emails
    if phones:
        meta["phones"] = phones

    # Identifiers
    if vats:
        meta["vat_ids"] = vats

    # Security
    pii_list = sorted(set(
        (["EMAIL"] if emails else []) +
        (["PHONE"] if phones else []) +
        (["IBAN"] if ibans else []) +
        presidio_entity_types
    ))
    if pii_list:
        meta["pii"] = pii_list

    # PII detections
    if presidio_findings:
        meta["pii_detections"] = presidio_findings
    # JSON-serializable normalization
    return json.loads(json.dumps(meta, ensure_ascii=False))
