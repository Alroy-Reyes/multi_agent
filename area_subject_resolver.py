import os
import re
import pandas as pd
from typing import Dict, List, Tuple

_CSV_CANDIDATES = [
    "/mnt/data/subject_area_override.csv",
    os.path.join(os.path.dirname(__file__), "subject_area_override.csv"),
]

def _load_csv_patterns() -> Dict[str, List[re.Pattern]]:
    for path in _CSV_CANDIDATES:
        if os.path.exists(path):
            df = pd.read_csv(path)
            pat_col = next((c for c in ["Pattern","pattern","Subject","subject","Name","name"] if c in df.columns), None)
            area_col = next((c for c in ["Area","area","SubjectArea","subject_area"] if c in df.columns), None)
            if not pat_col or not area_col:
                continue
            out: Dict[str, List[re.Pattern]] = {}
            for _, row in df.iterrows():
                pat = str(row[pat_col]).strip()
                area = str(row[area_col]).strip()
                if not pat or not area:
                    continue
                esc = re.escape(pat)
                esc = re.sub(r"\\\s+", r"\\s+", esc).replace("\\-", "[-\\s]?")
                out.setdefault(area, []).append(re.compile(rf"(?i)\b{esc}\b"))
            return out
    return {}

_CSV_PATTERNS = _load_csv_patterns()

_FALLBACK: Dict[str, List[str]] = {
    "Math": [
        r"\bbasic\s*calculus\b",
        r"\bpre[-\s]?calculus\b",
        r"\bstatistics(\s*(and|&)\s*probability)?\b",
        r"\bprobability\b",
        r"\bgeneral\s*mathematics\b|\bgen\s*math\b",
        r"\bbusiness\s*math\b",
    ],
    "English/Communication": [
        r"^\s*media\s+and\s+information\s+literacy\s*$",
        r"\boral\s*communication\b",
        r"\bread(ing)?\s*and\s*writ(ing)?\b",
        r"\beapp\b|\benglish\s*for\s*academic(\s*and)?\s*professional\s*purposes\b",
        r"\benglish\b",
        r"\bspeech\b",
        r"\bcommunication\b",
    ],
    "Science": [
        r"\bgeneral\s*chemistry\b|\bchemistry\b",
        r"\bgeneral\s*biology\b|\bbiology\b",
        r"\bgeneral\s*physics\b|\bphysics\b",
        r"\bearth\s*science\b",
        r"\bphysical\s*science\b",
        r"\benvironmental\s*science\b",
        r"\blaboratory\b",
    ],
    "PE/Health": [
        r"\bphysical\s*education\b|\b^pe\b",
        r"\bhealth\b",
        r"\bpersonal\s*development\b",
        r"\bfitness\b|\bsports\b",
    ],
    "ICT/Media": [
        r"\bict\b|\bprogramming\b|\bcomputer\s*science\b",
        r"\bcomputer\s*(systems|servicing)\b|\bcss\b",
        r"\banimation\b|\bmultimedia\b|\bweb\s*(dev(elopment)?|design)\b",
    ],
    "Business/ABM": [
        r"\baccounting\b|\bentrepreneurship\b|\bmarketing\b",
        r"\b(applied\s*)?economics\b",
        r"\borganization\s*and\s*management\b",
        r"\bbusiness\s*finance\b",
    ],
    "Filipino": [
        r"\bfilipino\b",
        r"\bkomunikasyon\b",
        r"\bpagbasa\b|\bpananaliksik\b|\bpagsusuri\b",
        r"\bkultura\b|\bwika\b",
    ],
    "Social Studies": [
        r"\bucsp\b|\bunderstanding\s*culture\s*society\s*and\s*politics\b",
        r"\bphilippine\s*politics\s*and\s*governance\b|\bppg\b",
        r"\b(philosophy|introduction\s*to\s*philosophy)\b",
        r"\bcontemporary\s*issues\b",
        r"\bworld\s*religions\b",
        r"\bdisciplines\s*and\s*ideas\b",
    ],
    "Literature": [
        r"\b21(st)?\s*century\s*literature\b",
        r"\bliterature\b",
        r"\bcreative\s*writing\b",
    ],
    "Arts/Design": [
        r"\bmedia\s*arts\b",
        r"\bdesign\b|\bvisual\s*arts\b|\bperforming\s*arts\b",
        r"\bcontemporary\s*philippine\s*arts\b|\bcpar\b",
    ],
    "Research": [
        r"\bpractical\s*research\s*(1|2)?\b",
        r"\bres(earch)?\s*project\b",
        r"\bquantitative\b|\bqualitative\b",
        r"\bcapstone\b",
    ],
}
_FALLBACK_COMPILED: Dict[str, List[re.Pattern]] = {
    area: [re.compile(rf"(?i){pat}") for pat in pats] for area, pats in _FALLBACK.items()
}

def assign_area(subject_name: str) -> str:
    s = (subject_name or "").strip()
    if not s:
        return "Other"
    # CSV first
    for area, patterns in _CSV_PATTERNS.items():
        if any(rx.search(s) for rx in patterns):
            return area
    # Fallbacks
    for area, patterns in _FALLBACK_COMPILED.items():
        if any(rx.search(s) for rx in patterns):
            return area
    return "Other"

def explain_area(subject_name: str) -> Tuple[str, str]:
    s = (subject_name or "").strip()
    if not s:
        return "Other", "none"
    for area, patterns in _CSV_PATTERNS.items():
        if any(rx.search(s) for rx in patterns):
            return area, "csv"
    for area, patterns in _FALLBACK_COMPILED.items():
        if any(rx.search(s) for rx in patterns):
            return area, "fallback"
    return "Other", "none"
