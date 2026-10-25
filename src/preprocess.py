import html
import json
import re
from pathlib import Path

import emoji
from nltk.stem.isri import ISRIStemmer
from tashaphyne.stemming import ArabicLightStemmer

RESOURCES = Path(__file__).parent / "resources"

URL_TOKEN = "<URL>"
USER_TOKEN = "<USER>"
PLACEHOLDERS = {URL_TOKEN, USER_TOKEN}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


def load_mapping(name):
    return json.loads((RESOURCES / name).read_text(encoding="utf-8"))


DIALECT_TO_MSA = load_mapping("dialect_to_msa.json")
HYPERNYMS = load_mapping("hypernyms.json")

isri = ISRIStemmer()
light = ArabicLightStemmer()


def map_words(text, mapping):
    return " ".join(mapping.get(word, word) for word in text.split())


def apply_per_word(text, func):
    """Apply func to every word, leaving <URL>/<USER> placeholders untouched."""
    return " ".join(w if w in PLACEHOLDERS else func(w) for w in text.split())


def light_stem_word(word):
    light.light_stem(word)
    return light.get_stem()


def clean_html(text):
    return re.sub(r"<[^>]+>", " ", html.unescape(text))


def remove_urls(text):
    return URL_PATTERN.sub(" ", text)


def replace_urls(text):
    return URL_PATTERN.sub(f" {URL_TOKEN} ", text)


def replace_mentions(text):
    return re.sub(r"@\w+", f" {USER_TOKEN} ", text)


def segment_hashtags(text):
    return re.sub(r"#(\w+)", lambda m: " " + m.group(1).replace("_", " ") + " ", text)


def emoji_to_arabic(text):
    return emoji.demojize(text, language="ar", delimiters=(" ", " ")).replace("_", " ")


def remove_diacritics(text):
    return re.sub(r"[\u064B-\u0652]", "", text)


def remove_kashida(text):
    return text.replace("\u0640", "")


def normalize_letters(text):
    return re.sub(r"[إأآ]", "ا", text).replace("ى", "ي").replace("ة", "ه")


def reduce_elongation(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def remove_digits(text):
    return re.sub(r"[0-9\u0660-\u0669]+", " ", text)


def remove_punctuation(text):
    # Keeps placeholders, and keeps diacritics/kashida so those steps stay independent.
    return re.sub(
        r"<(?:URL|USER)>|[^\w\s\u064B-\u0652\u0640]",
        lambda m: m.group(0) if m.group(0).startswith("<") else " ",
        text,
    )


def dialect_to_msa(text):
    return map_words(text, DIALECT_TO_MSA)


def hypo_to_hyper(text):
    return map_words(text, HYPERNYMS)


def light_stem(text):
    return apply_per_word(text, light_stem_word)


def stem(text):
    return apply_per_word(text, isri.stem)


# Order here IS the execution order, regardless of the order of names passed in.
STEPS = {
    "html": clean_html,
    "remove_urls": remove_urls,        # choose at most one of these two;
    "replace_urls": replace_urls,      # if neither is chosen, URLs are kept
    "mentions": replace_mentions,
    "hashtags": segment_hashtags,
    "emoji": emoji_to_arabic,
    "diacritics": remove_diacritics,
    "kashida": remove_kashida,
    "letters": normalize_letters,
    "elongation": reduce_elongation,
    "digits": remove_digits,
    "punctuation": remove_punctuation,
    "dialect": dialect_to_msa,
    "hypernyms": hypo_to_hyper,
    "light_stem": light_stem,
    "stem": stem,
}


def build_pipeline(names):
    if isinstance(names, str):
        names = [names]
    names = set(names)

    unknown = names - STEPS.keys()
    if unknown:
        raise ValueError(f"Unknown preprocessing steps: {sorted(unknown)}. "
                         f"Valid: {list(STEPS)}")
    if {"remove_urls", "replace_urls"} <= names:
        raise ValueError("Choose only one of 'remove_urls' or 'replace_urls'.")

    steps = [func for name, func in STEPS.items() if name in names]

    def run(text):
        for step in steps:
            text = step(text)
        return " ".join(text.split())

    return run