"""Letters, language tags and sentence splitting (same rules as the original service).

normalize()        = the original character normalization (lower case, a few replacements, accents except ä ö ü ß removed)
letters()          = ids + language per letter, with [en]...[/en] / [de]...[/de] switches
split_sentences()  = server.saetze: finished sentences from a growing buffer; short ones wait, long ones split at commas
"""
from __future__ import annotations

import re
import unicodedata

CHARSET = " abcdefghijklmnopqrstuvwxyzäöüß.,?!-'\":;()0123456789"
ID = {c: i + 1 for i, c in enumerate(CHARSET)}          # 0 = padding / prompt placeholder
VOCAB = len(CHARSET) + 1
LANGUAGES = {"de": 0, "en": 1}
REPLACE = {"’": "'", "‘": "'", "`": "'", "´": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
           "–": "-", "—": "-", "…": ".", "ø": "ö", "æ": "ä", "ð": "d", "þ": "th"}
IS_LETTER = [False] + [c.isalnum() for c in CHARSET]   # letters and digits count for the speaking rate

# Language switches survive the German frontend as private-use characters.
TAG = {"[en]": "", "[/en]": "", "[de]": "", "[/de]": ""}
_TAG_BACK = {v: k for k, v in TAG.items()}
_TAG_RX = re.compile(r"\[(/?)(de|en)\]", re.I)


def protect_tags(text: str) -> str:
    return _TAG_RX.sub(lambda m: TAG[f"[{m.group(1)}{m.group(2).lower()}]"], text)


def normalize(text: str) -> str:
    out = []
    for c in text.lower():
        c = REPLACE.get(c, c)
        if c not in ID and c not in "äöüß":
            c = unicodedata.normalize("NFKD", c)
            c = "".join(x for x in c if not unicodedata.combining(x))
        for x in c:
            if x in ID:
                out.append(x)
            elif x.isspace():
                out.append(" ")
    return " ".join("".join(out).split())


def letters(text: str, language: str = "de") -> tuple[list[int], list[int]]:
    """-> (letter ids, language id per letter). [en]..[/en] (or the protected form) switches the language."""
    text = protect_tags(text)
    space = ID[" "]
    ids: list[int] = []
    lang: list[int] = []
    current = language
    for part in re.split("([-])", text):
        if part in _TAG_BACK:
            tag = _TAG_BACK[part]
            current = language if tag.startswith("[/") else tag[1:3]
            continue
        n = normalize(part)
        if not n:
            continue
        if ids and part[:1].isspace() and ids[-1] != space:
            ids.append(space)
            lang.append(LANGUAGES[current])
        ids += [ID[c] for c in n]
        lang += [LANGUAGES[current]] * len(n)
        if part[-1:].isspace():
            ids.append(space)
            lang.append(LANGUAGES[current])
    while ids and ids[-1] == space:
        ids.pop()
        lang.pop()
    return ids, lang


MAX_CHARS = 180          # with the voice prompt in front, the model has room for about this many characters
MIN_LETTERS = 20         # shorter sentences wait for the next one
_SENTENCE = re.compile(r"(.+?[.!?…:;]+[\"')\]]*)(?:\s+|$)", re.S)


def split_sentences(buffer: str, final: bool) -> tuple[list[str], str]:
    """Finished sentences from the buffer -> (sentences, rest). final=True flushes everything."""
    out, pos = [], 0
    for m in _SENTENCE.finditer(buffer):
        if m.end() == len(buffer) and not final and not buffer[-1:].isspace():
            break                                   # punctuation at the very end: "..." or a number may follow
        out.append(m.group(1).strip())
        pos = m.end()
    rest = buffer[pos:]
    if final and rest.strip():
        out.append(rest.strip())
        rest = ""
    joined: list[str] = []
    for s in out:                                   # short sentences are joined to the previous one
        if joined and sum(c.isalpha() for c in joined[-1]) < MIN_LETTERS:
            joined[-1] += " " + s
        else:
            joined.append(s)
    if joined and not final and sum(c.isalpha() for c in joined[-1]) < MIN_LETTERS:
        rest = joined.pop() + " " + rest
    done = []
    for s in joined:                                # very long sentences are split at a comma (or a space)
        while len(s) > MAX_CHARS:
            k = s.rfind(", ", 60, MAX_CHARS)
            k = k + 1 if k > 0 else s.rfind(" ", 60, MAX_CHARS)
            if k <= 0:
                break
            done.append(s[:k].strip())
            s = s[k:].strip()
        done.append(s)
    return done, rest
