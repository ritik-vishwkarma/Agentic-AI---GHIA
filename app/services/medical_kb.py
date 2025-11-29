"""Medical Knowledge Base for symptom normalization and mapping"""

from typing import Dict, List, Tuple
import re

# Hinglish to Medical English Mapping
# This lexicon maps colloquial Hindi/Hinglish phrases to standardized medical terms
SYMPTOM_LEXICON: Dict[str, str] = {
    # Pain related
    "dard": "pain",
    "dard ho raha hai": "pain",
    "bahut dard": "severe pain",
    "bhut dard": "severe pain",
    "halka dard": "mild pain",
    "thoda dard": "mild pain",
    "tej dard": "sharp pain",
    "dhimi dard": "dull aching pain",
    # Body parts
    "kamar": "lower back",
    "kamar me dard": "lower back pain",
    "sir": "head",
    "sir dard": "headache",
    "sar dard": "headache",
    "pet": "abdomen",
    "pet me dard": "abdominal pain",
    "pet dard": "stomach pain",
    "seena": "chest",
    "chhati": "chest",
    "chest me dard": "chest pain",
    "gala": "throat",
    "gale me dard": "sore throat",
    "kaan": "ear",
    "kaan me dard": "ear pain",
    "aankh": "eye",
    "aankho me": "in eyes",
    "haath": "hand/arm",
    "pair": "leg/foot",
    "ghutna": "knee",
    "ghutne me dard": "knee pain",
    "kandha": "shoulder",
    "peeth": "back",
    # Fever related
    "bukhar": "fever",
    "tez bukhar": "high fever",
    "halka bukhar": "low grade fever",
    "badan garam": "body feels hot",
    "thandi lag rahi": "feeling chills",
    "kaanpna": "shivering",
    # Respiratory
    "khansi": "cough",
    "khansi aa rahi": "having cough",
    "sukhi khansi": "dry cough",
    "balgam": "phlegm/mucus",
    "balgam wali khansi": "productive cough",
    "saans": "breath/breathing",
    "saans lene me dikkat": "difficulty breathing",
    "saans phoolna": "shortness of breath",
    "ghar ghar": "wheezing sound",
    "saans ghar ghar": "wheezing",
    "naak band": "nasal congestion",
    "naak beh rahi": "runny nose",
    "nazla": "cold/rhinitis",
    "zukam": "cold",
    # GI symptoms
    "ulti": "vomiting",
    "ji machlana": "nausea",
    "matli": "nausea",
    "dast": "diarrhea",
    "loose motion": "loose stools",
    "kabz": "constipation",
    "pet me gas": "bloating/gas",
    "khana hazam nahi": "indigestion",
    "bhookh nahi": "loss of appetite",
    "acidity": "heartburn/acid reflux",
    "khatti dakar": "acid reflux",
    # General symptoms
    "kamzori": "weakness",
    "thakaan": "fatigue",
    "chakkar": "dizziness",
    "chakkar aa raha": "feeling dizzy",
    "behoshi": "fainting",
    "neend nahi": "insomnia",
    "sojah": "swelling",
    "sujan": "swelling",
    "laal": "redness",
    "khujli": "itching",
    "daane": "rash",
    "paseena": "sweating",
    "zyada paseena": "excessive sweating",
    # Sensations
    "gudgudi": "tingling sensation",
    "sunn": "numbness",
    "sunn pad gaya": "numbness",
    "jalan": "burning sensation",
    "chubhan": "pricking sensation",
    # Duration phrases
    "kal se": "since yesterday",
    "do din se": "for 2 days",
    "teen din se": "for 3 days",
    "ek hafta se": "for 1 week",
    "ek hafte se": "for 1 week",
    "mahine se": "for a month",
    "kai din se": "for several days",
    # Severity
    "bahut zyada": "very severe",
    "thoda": "mild",
    "halka": "mild",
    "tej": "severe/sharp",
    "bardasht nahi": "unbearable",
    # Urinary
    "peshab me jalan": "burning urination",
    "bar bar peshab": "frequent urination",
    "peshab kam": "reduced urination",
    # Women's health
    "mahwari": "menstruation",
    "periods": "menstruation",
    "zyada bleeding": "heavy menstrual bleeding",
    "dard mahwari": "menstrual cramps",
}

# Red flag symptoms that need urgent attention
RED_FLAG_SYMPTOMS = [
    "chest pain",
    "difficulty breathing",
    "severe chest pain",
    "unconscious",
    "fainting",
    "seizure",
    "severe bleeding",
    "high fever",
    "severe headache",
    "numbness on one side",
    "slurred speech",
    "sudden vision loss",
    "severe abdominal pain",
    "blood in stool",
    "blood in vomit",
]

# Symptom categories for structured extraction
SYMPTOM_CATEGORIES = {
    "pain": ["pain", "dard", "ache", "hurt"],
    "fever": ["fever", "bukhar", "temperature", "hot"],
    "respiratory": ["cough", "breath", "saans", "wheeze", "khansi"],
    "gi": ["vomit", "diarrhea", "nausea", "stomach", "pet"],
    "neurological": ["headache", "dizzy", "numb", "tingling", "chakkar"],
    "general": ["weak", "tired", "fatigue", "thakan"],
}


def get_symptom_severity_indicators() -> Dict[str, List[str]]:
    """
    Returns severity indicator keywords for symptom classification.
    Used by agents to assess symptom severity from patient descriptions.
    """
    return {
        "severe": [
            "bahut",
            "बहुत",
            "severe",
            "extreme",
            "unbearable",
            "असहनीय",
            "terrible",
            "worst",
            "intense",
            "तेज",
            "jyada",
            "ज्यादा",
            "serious",
            "critical",
            "emergency",
            "खतरनाक",
            "dangerous",
        ],
        "moderate": [
            "medium",
            "मध्यम",
            "moderate",
            "thoda",
            "थोड़ा",
            "some",
            "manageable",
            "bearable",
            "normal",
            "average",
            "सामान्य",
        ],
        "mild": [
            "halka",
            "हल्का",
            "mild",
            "slight",
            "little",
            "minor",
            "kam",
            "कम",
            "light",
            "थोड़ा सा",
            "barely",
        ],
    }


def normalize_hinglish_text(raw_text: str) -> Tuple[str, List[Dict]]:
    """
    Normalize Hinglish text by mapping colloquial phrases to medical terms.

    Args:
        raw_text: Raw transcribed text in Hindi/Hinglish

    Returns:
        Tuple of (normalized_text, list of mapped phrases)
    """
    text = raw_text.lower().strip()

    # Basic cleaning - keep Hindi Unicode range
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    mapped_phrases = []

    # Sort by phrase length (longer phrases first) to avoid partial matches
    sorted_lexicon = sorted(
        SYMPTOM_LEXICON.items(), key=lambda x: len(x[0]), reverse=True
    )

    for phrase, mapped_term in sorted_lexicon:
        if phrase in text:
            mapped_phrases.append(
                {
                    "original": phrase,
                    "mapped_to": mapped_term,
                    "category": _get_symptom_category(mapped_term),
                }
            )
            # Replace in text
            text = text.replace(phrase, mapped_term)

    return text, mapped_phrases


def _get_symptom_category(term: str) -> str:
    """Get category for a symptom term"""
    term_lower = term.lower()
    for category, keywords in SYMPTOM_CATEGORIES.items():
        if any(kw in term_lower for kw in keywords):
            return category
    return "other"


def check_red_flags(symptoms: List[str]) -> List[str]:
    """
    Check if any symptoms are red flags requiring urgent attention.

    Args:
        symptoms: List of symptom strings

    Returns:
        List of red flag symptoms found
    """
    found_red_flags = []
    symptoms_text = " ".join(symptoms).lower()

    for red_flag in RED_FLAG_SYMPTOMS:
        if red_flag.lower() in symptoms_text:
            found_red_flags.append(red_flag)

    return found_red_flags


def extract_duration(text: str) -> str:
    """Extract duration information from text"""
    duration_patterns = [
        (r"(\d+)\s*(din|day|days)", "{} days"),
        (r"(\d+)\s*(hafte|hafta|week|weeks)", "{} weeks"),
        (r"(\d+)\s*(mahine|mahina|month|months)", "{} months"),
        (r"kal se|since yesterday", "1 day"),
        (r"aaj se|today", "today"),
        (r"kai din|several days", "several days"),
    ]

    text_lower = text.lower()

    for pattern, template in duration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if match.groups():
                return template.format(match.group(1))
            else:
                return template

    return None
