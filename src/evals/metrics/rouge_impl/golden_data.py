"""
Golden (target, prediction) pairs for ROUGE parity tests and benchmarks.

Shared by tests and benchmark_rouge.py so the script can run without the tests package.
"""

# (target, prediction) = (ground_truth, generated)
ROUGE_GOLDEN_PAIRS: list[tuple[str, str]] = [
    ("", ""),
    ("", "hello"),
    ("hello", ""),
    ("the cat sat on the mat", "the cat sat"),
    ("the cat sat on the mat", "the cat sat on the mat"),
    ("running runner runs", "run runner running"),
    ("evaluation evaluated evaluates", "evalu evaluation"),
    ("hello world", "hello world"),
    ("Hello World", "HELLO WORLD"),
    ("a b c", "a b c"),
    ("a b c", "x y z"),
    ("one two three four", "one two three"),
    ("1 2 3", "1 2 3"),
    ("test123 token", "test123 token"),
    ("The quick brown fox jumps over the lazy dog.", "The quick brown dog jumps on the log."),
    ("Machine learning is a subset of artificial intelligence.", "Machine learning is part of AI."),
    ("Natural language processing enables computers to understand text.", "NLP helps computers understand language."),
    ("the the the", "the the"),
    ("a", "a"),
    ("ab cd ef", "ab cd ef"),
    ("evaluation", "evaluate"),
    ("running", "run"),
    ("running", "running"),
    ("computers", "computer"),
    ("studies", "study"),
    ("studied", "study"),
    ("evaluating", "evaluate"),
    ("lowercase UPPERCASE MiXed", "lowercase uppercase mixed"),
    ("punctuation! period. comma, semi; colon:", "punctuation period comma semi colon"),
    ("extra   spaces   here", "extra spaces here"),
    ("tab\tseparated", "tab separated"),
    ("line1\nline2", "line1 line2"),
    ("dot.dot", "dot dot"),
    ("hyphen-word", "hyphen word"),
    ("apostrophe's", "apostrophe s"),
    ("num42ber", "num 42 ber"),
    ("café", "cafe"),
    ("naïve", "naive"),
    ("résumé", "resume"),
    ("long " + "word " * 20, "long " + "word " * 15),
    ("short", "short"),
    ("a b a b a", "a b a b"),
    ("x y z", "x y z"),
    ("alpha beta gamma", "alpha beta"),
    ("first second third fourth fifth", "first second third"),
    ("repeated repeated repeated", "repeated repeated"),
    ("unique words here", "unique words there"),
    ("the end", "the end"),
    ("start the", "start the"),
    ("one", "one"),
    ("two words", "two words"),
    ("three word phrase", "three word phrase"),
    ("four word phrase here", "four word phrase"),
    ("five word phrase right here", "five word phrase right"),
    ("six word phrase right over here", "six word phrase right over"),
    ("seven word phrase right over here now", "seven word phrase right over here"),
    ("eight word phrase right over here now then", "eight word phrase right over here now"),
    ("nine word phrase right over here now then again", "nine word phrase right over here now then"),
    ("ten word phrase right over here now then again so", "ten word phrase right over here now then again"),
]
for i in range(20):
    ROUGE_GOLDEN_PAIRS.append(
        (f"sentence {i} with some content.", f"sentence {i} with content.")
    )
for i in range(20):
    ROUGE_GOLDEN_PAIRS.append(
        (f"token{i} token{i+1} token{i+2}", f"token{i} token{i+1}")
    )


def get_golden_gen_gt_lists() -> tuple[list[str], list[str]]:
    """Return (gen_list, gt_list) from golden pairs (prediction, target)."""
    gen_list = [p for _, p in ROUGE_GOLDEN_PAIRS]
    gt_list = [t for t, _ in ROUGE_GOLDEN_PAIRS]
    return gen_list, gt_list
