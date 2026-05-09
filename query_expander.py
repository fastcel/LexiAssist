"""
query_expander.py — Rule-based query expansion for improved retrieval recall.

No external API calls. Generates sub-queries using three hand-built strategies:

  1. Synonym substitution  — swaps plain-English terms for legal equivalents
                             and vice versa (bidirectional legal thesaurus)
  2. Keyword extraction    — strips stopwords and question words, keeping only
                             the content terms as a compact keyword query
  3. Structural rephrasing — rewrites question form into noun-phrase form
                             (e.g. "What is X?" → "definition of X")

All three strategies are applied to every query and deduplicated.
The caller receives up to n sub-queries to run alongside the original.
"""

import re


# ── Legal synonym map (bidirectional) ─────────────────────────────────────────
# Keys are plain-English terms; values are legal equivalents.
# Expansion is applied in both directions so a user typing legalese also
# gets plain-English sub-queries that may match differently-worded chunks.

LEGAL_SYNONYMS: dict[str, list[str]] = {
    # relationships / family
    "defacto":          ["de facto", "domestic partnership", "cohabitation", "de facto relationship"],
    "de facto":         ["defacto", "domestic partnership", "cohabitation"],
    "divorce":          ["dissolution of marriage", "matrimonial breakdown", "separation"],
    "separation":       ["matrimonial breakdown", "divorce", "estrangement"],
    "custody":          ["parental responsibility", "care arrangements", "child custody orders"],
    "child support":    ["maintenance", "financial support for children", "child maintenance"],

    # property / contract
    "landlord":         ["lessor", "property owner", "licensor"],
    "tenant":           ["lessee", "occupant", "renter"],
    "lease":            ["tenancy agreement", "rental agreement", "licence to occupy"],
    "eviction":         ["termination of tenancy", "notice to vacate", "unlawful eviction"],
    "contract":         ["agreement", "deed", "instrument", "covenant"],
    "breach":           ["violation", "failure to comply", "contravention", "non-performance"],
    "damages":          ["compensation", "remedy", "loss", "restitution"],
    "negligence":       ["duty of care", "tort", "carelessness", "liability"],
    "liable":           ["responsible", "at fault", "negligent", "culpable"],

    # rights / process
    "rights":           ["entitlements", "legal rights", "provisions", "protections"],
    "law":              ["legislation", "act", "statute", "regulation", "legal provisions"],
    "court":            ["tribunal", "judicial body", "magistrates court", "federal court"],
    "sue":              ["commence proceedings", "bring a claim", "litigate", "seek damages"],
    "fine":             ["penalty", "infringement notice", "pecuniary penalty"],
    "appeal":           ["review", "challenge", "appellate proceedings"],
    "consent":          ["agreement", "permission", "authorisation", "approval"],
    "terminate":        ["end", "cancel", "rescind", "discharge"],
    "void":             ["invalid", "unenforceable", "null and void"],

    # criminal
    "crime":            ["offence", "criminal act", "indictable offence"],
    "accused":          ["defendant", "respondent", "charged person"],
    "guilty":           ["convicted", "found liable", "culpable"],
    "sentence":         ["penalty", "punishment", "order", "sanction"],

    # corporate
    "company":          ["corporation", "body corporate", "incorporated entity"],
    "director":         ["officer", "board member", "company director"],
    "shareholder":      ["member", "equity holder", "stockholder"],
    "insolvency":       ["bankruptcy", "liquidation", "winding up"],
}

# Build reverse map (legal term → plain English)
_REVERSE: dict[str, list[str]] = {}
for plain, legals in LEGAL_SYNONYMS.items():
    for legal in legals:
        _REVERSE.setdefault(legal, []).append(plain)

# Merge both directions into one lookup
_ALL_SYNONYMS: dict[str, list[str]] = {**LEGAL_SYNONYMS}
for term, plains in _REVERSE.items():
    _ALL_SYNONYMS.setdefault(term, []).extend(plains)


# ── Stopwords ─────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "their", "this", "that", "these", "those", "and", "or", "but", "if",
    "in", "on", "at", "to", "for", "of", "with", "about", "under", "by",
    "from", "into", "through", "during", "before", "after", "between",
    "not", "no", "so", "as", "up", "out", "there", "then", "than",
    "any", "all", "both", "each", "few", "more", "most", "other", "some",
}

_QUESTION_STARTERS = re.compile(
    r"^\s*(what|how|when|where|who|why|is|are|can|does|do|did|will|"
    r"what is|what are|how does|how do|can i|am i)\s+",
    re.IGNORECASE,
)


# ── Strategy 1: synonym substitution ─────────────────────────────────────────

def _synonym_variants(query: str) -> list[str]:
    """Replace each recognised term with its first synonym to form a new query."""
    variants = []
    words = query.lower().split()

    for i, word in enumerate(words):
        clean = word.strip("?.,;:'\"")
        if clean in _ALL_SYNONYMS:
            for synonym in _ALL_SYNONYMS[clean][:2]:   # at most 2 replacements per term
                new_words = words[:i] + [synonym] + words[i + 1:]
                variant = " ".join(new_words).strip("?. ")
                if variant.lower() != query.lower():
                    variants.append(variant)

    # also try two-word phrases
    for i in range(len(words) - 1):
        phrase = f"{words[i].strip('?.,;:')} {words[i+1].strip('?.,;:')}"
        if phrase in _ALL_SYNONYMS:
            synonym = _ALL_SYNONYMS[phrase][0]
            new_words = words[:i] + [synonym] + words[i + 2:]
            variant = " ".join(new_words).strip("?. ")
            if variant.lower() != query.lower():
                variants.append(variant)

    return variants


# ── Strategy 2: keyword extraction ───────────────────────────────────────────

def _keyword_query(query: str) -> str:
    """Strip stopwords and question words; return content terms only."""
    # remove question starters
    stripped = _QUESTION_STARTERS.sub("", query).strip("?. ")
    tokens = stripped.lower().split()
    keywords = [t.strip("?.,;:'\"") for t in tokens if t.strip("?.,;:'\"") not in _STOPWORDS]
    return " ".join(keywords)


# ── Strategy 3: structural rephrasing ────────────────────────────────────────

_REPHRASE_PATTERNS = [
    # "What is X?" → "definition of X"
    (re.compile(r"^what\s+is\s+(.+?)\??$", re.I),       r"definition of \1"),
    # "What are X?" → "types of X"
    (re.compile(r"^what\s+are\s+(.+?)\??$", re.I),       r"types of \1"),
    # "How does X work?" → "X process"
    (re.compile(r"^how\s+does\s+(.+?)\s+work\??$", re.I), r"\1 process"),
    # "Can I X?" → "right to X"
    (re.compile(r"^can\s+i\s+(.+?)\??$", re.I),           r"right to \1"),
    # "Am I entitled to X?" → "entitlement to X"
    (re.compile(r"^am\s+i\s+entitled\s+to\s+(.+?)\??$", re.I), r"entitlement to \1"),
    # "What happens if X?" → "consequences of X"
    (re.compile(r"^what\s+happens\s+if\s+(.+?)\??$", re.I), r"consequences of \1"),
    # "Do I have to X?" → "obligation to X"
    (re.compile(r"^do\s+i\s+have\s+to\s+(.+?)\??$", re.I), r"obligation to \1"),
    # "What are my rights regarding X?" → "rights X"
    (re.compile(r"^what\s+are\s+my\s+rights\s+(?:regarding|about|for|in)\s+(.+?)\??$", re.I), r"legal rights \1"),
]


def _structural_rephrase(query: str) -> list[str]:
    """Apply regex rewrite rules to produce noun-phrase forms of the query."""
    results = []
    for pattern, template in _REPHRASE_PATTERNS:
        m = pattern.match(query.strip())
        if m:
            results.append(pattern.sub(template, query.strip()))
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def expand_query(query: str, n: int = 3) -> list[str]:
    """
    Generate up to n sub-queries from the original query using rule-based methods.

    Three strategies are applied in priority order:
      1. Synonym substitution (legal thesaurus, bidirectional)
      2. Structural rephrasing (question → noun phrase)
      3. Keyword extraction (stopword removal)

    Args:
        query: Original user question.
        n:     Maximum number of sub-queries to return.

    Returns:
        List of up to n distinct sub-query strings (may be fewer if the query
        is short or doesn't match any patterns). Always safe to call — never
        raises exceptions.
    """
    seen = {query.lower().strip()}
    candidates: list[str] = []

    # strategy 1: synonyms
    for v in _synonym_variants(query):
        if v.lower() not in seen and v.strip():
            seen.add(v.lower())
            candidates.append(v)

    # strategy 2: structural rephrase
    for v in _structural_rephrase(query):
        if v.lower() not in seen and v.strip():
            seen.add(v.lower())
            candidates.append(v)

    # strategy 3: keyword query (always useful as a fallback)
    kw = _keyword_query(query)
    if kw and kw.lower() not in seen and len(kw.split()) >= 2:
        seen.add(kw.lower())
        candidates.append(kw)

    return candidates[:n]