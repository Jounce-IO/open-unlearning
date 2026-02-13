"""
ROUGE tokenizer matching rouge_score behavior exactly.

Lowercase -> replace [^a-z0-9]+ with space -> split on whitespace ->
optional Porter stem (only len>3) -> filter ^[a-z0-9]+$.
"""

import re

# Pre-compile regexes to match rouge_score
NON_ALPHANUM_RE = re.compile(r"[^a-z0-9]+")
SPACES_RE = re.compile(r"\s+")
VALID_TOKEN_RE = re.compile(r"^[a-z0-9]+$")


def tokenize(text: str, use_stemmer: bool) -> list:
    """
    Tokenize input text into a list of tokens (rouge_score spec).

    Args:
        text: Input text.
        use_stemmer: If True, apply NLTK Porter stemmer to tokens with len > 3.

    Returns:
        List of string tokens.
    """
    text = text.lower()
    text = NON_ALPHANUM_RE.sub(" ", text)
    tokens = SPACES_RE.split(text)
    if use_stemmer:
        from nltk.stem import porter
        stemmer = porter.PorterStemmer()
        tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]
    tokens = [x for x in tokens if VALID_TOKEN_RE.match(x)]
    return tokens
