import html
import json
import re
from pathlib import Path

import emoji
from nltk.stem.isri import ISRIStemmer
from tashaphyne.stemming import ArabicLightStemmer

RESOURCES = Path(__file__).parent / "resources"


def load_mapping(name):
    return json.loads((RESOURCES / name).read_text(encoding="utf-8"))


DIALECT_TO_MSA = load_mapping("dialect_to_msa.json")
HYPERNYMS = load_mapping("hypernyms.json")

isri = ISRIStemmer()
light = ArabicLightStemmer()


def map_words(text, mapping):
    return " ".join(mapping.get(word, word) for word in text.split())


def light_stem_word(word):
    light.light_stem(word)
    return light.get_stem()


def clean_html(text):
    return re.sub(r"<[^>]+>", " ", html.unescape(text))


def replace_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", " <URL> ", text)


def replace_mentions(text):
    return re.sub(r"@\w+", " <USER> ", text)


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


def dialect_to_msa(text):
    return map_words(text, DIALECT_TO_MSA)


def hypo_to_hyper(text):
    return map_words(text, HYPERNYMS)


def light_stem(text):
    return " ".join(light_stem_word(word) for word in text.split())


def stem(text):
    return " ".join(isri.stem(word) for word in text.split())

def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", " ", text)

STEPS = {
    "html": clean_html,
    "urls": replace_urls,
    "mentions": replace_mentions,
    "hashtags": segment_hashtags,
    "emoji": emoji_to_arabic,
    "diacritics": remove_diacritics,
    "kashida": remove_kashida,
    "letters": normalize_letters,
    "elongation": reduce_elongation,
    "digits": remove_digits,
    "dialect": dialect_to_msa,
    "hypernyms": hypo_to_hyper,
    "light_stem": light_stem,
    "stem": stem,
}


def build_pipeline(names):
    steps = [func for name, func in STEPS.items() if name in names]

    def run(text):
        for step in steps:
            text = step(text)
        return " ".join(text.split())

    return run
