"""German text frontend: turns written text (as a language model writes it) into words the model can read.

The model reads letters and never saw digits during training, so everything that is not plain words has to be
written out first. :func:`normalize` runs these stages, in this order:

  phone_numbers        +49 170 1234567 -> plus vier neun, eins sieben null, ...
  clean_markup         Markdown, LaTeX, emoji, smileys, arrows, list numbers, e-mail and web addresses
  expand_units         230 V -> 230 Volt, 1 h -> eine Stunde (units.json)
  math_symbols         a², √2, 2^10, 5 + 3, 10 - 4
  numbers              dates, times, money, fractions, ordinals, decimals, plain numbers (num2words)
  expand_abbreviations z. B. -> zum Beispiel (abbreviations.json)
  spell_out_letters    PDF -> peh deh eff, single letters and vowel-less abbreviations

:class:`StreamNormalizer` releases a token stream sentence by sentence, without cutting through abbreviations,
list numbers or ordinals at a chunk border.

This is a port of the original frontend the model was evaluated with (14 October 2026), with English names; its
behaviour, including the word lists and both JSON tables, is unchanged (checked against the original during development).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from num2words import num2words

DATA = Path(__file__).resolve().parent / "data"

# ------------------------------------------------------------------------------------------------ abbreviations
#
# Written out before synthesis: the model reads "z. B." letter by letter or swallows it. Writing it out also keeps
# the period of "z. B." from looking like the end of a sentence.

# Characters held back at the end of a stream buffer, because half an abbreviation could still be there
# ("z." in one token, " B." in the next): the longest abbreviation plus a margin.
HOLD_BACK = 16

# If an abbreviation ends with a period, that period was often also the end of the sentence ("im 3. Jh. v. Chr. Er
# zeigte"); written out, the pause would be lost. The period stays if a typical sentence start follows. A capital
# letter alone is not enough: every German noun starts with one.
_SENTENCE_START = re.compile(r"\s+(?:Er|Sie|Es|Der|Die|Das|Den|Dem|Ein|Eine|Einen|Danach|Dann|"
                             r"Später|Damals|Im|In|Am|Auch|Doch|Aber|Wir|Ich|Man|Dies|Diese|Dieser|"
                             r"Dieses|So|Heute|Seit|Bis|Nach|Vor|Als|Wenn|Weil|Hier|Dort|Zudem|"
                             r"Außerdem|Deshalb|Daher|Trotzdem|Jedoch|Allerdings)\b")


def _keep_period(replacement: str):
    def f(m: re.Match) -> str:
        # also at the end of the text: the period was the end of the sentence as well
        end = not m.string[m.end():].strip()
        return replacement + ("." if end or _SENTENCE_START.match(m.string, m.end()) else "")
    return f


_ABBREVIATION_RULES: list | None = None
_ACRONYM_WORDS: dict[str, str] = {}        # all-caps entries of the table (RAM -> Ramm), also for plurals ("RAMs")


def _abbreviation_rules() -> list:
    """Compiled (pattern, replacement) pairs: abbreviations first (longest first), then symbols."""
    global _ABBREVIATION_RULES
    if _ABBREVIATION_RULES is not None:
        return _ABBREVIATION_RULES
    data = json.loads((DATA / "abbreviations.json").read_text(encoding="utf-8"))
    rules = []
    # Longest first, otherwise "Mi." would match before "Mio." and leave the "o." behind.
    for k, v in sorted(data.get("abbreviations", {}).items(), key=lambda x: -len(x[0])):
        pat = re.escape(k).replace("\\ ", r"\s*")
        # no letter before and none after, otherwise "Str." would also match inside a word
        rules.append((re.compile(r"(?<![\w\u00c0-\u024f])" + pat + r"(?![\w\u00c0-\u024f])"),
                      _keep_period(v) if k.endswith(".") else v))
    symbols = [(re.compile(re.escape(k)), v) for k, v in
               sorted(data.get("symbols", {}).items(), key=lambda x: -len(x[0]))]
    _ACRONYM_WORDS.clear()
    _ACRONYM_WORDS.update({k: v for k, v in data.get("abbreviations", {}).items()
                           if re.fullmatch(r"[A-ZÄÖÜ]{2,6}", k)})
    _ABBREVIATION_RULES = rules + symbols
    return _ABBREVIATION_RULES


# "Mo–Fr" is a range, not two abbreviations with a dash. The period after the second one stays if a sentence
# follows or ends.
_WEEKDAY_ABBR = {"Mo": "Montag", "Di": "Dienstag", "Mi": "Mittwoch", "Do": "Donnerstag",
                 "Fr": "Freitag", "Sa": "Samstag", "So": "Sonntag"}
_WEEKDAY_RANGE = re.compile(r"\b(Mo|Di|Mi|Do|Fr|Sa|So)\.?[ \t]*[–-][ \t]*(Mo|Di|Mi|Do|Fr|Sa|So)\b"
                            r"(?:\.(?=[ \t]*[\da-zäöüß]))?")


def expand_abbreviations(text: str) -> str:
    """Write out abbreviations and symbols from abbreviations.json (z. B. -> zum Beispiel, % -> Prozent)."""
    if not text:
        return text
    text = _WEEKDAY_RANGE.sub(lambda m: f"{_WEEKDAY_ABBR[m.group(1)]} bis {_WEEKDAY_ABBR[m.group(2)]}", text)
    for rx, replacement in _abbreviation_rules():
        text = rx.sub(replacement if callable(replacement) else replacement.replace("\\", "\\\\"), text)
    return re.sub(r"[ \t]{2,}", " ", text)


# ------------------------------------------------------------------------------------------------ spelling out
#
# The model reads letter by letter after lower-casing: "PDF" becomes the made-up word "pdf", "KI" becomes "key". So
# upper-case abbreviations are written out with German letter names. Abbreviations spoken as a word (NASA, NATO,
# TÜV, PIN) are in abbreviations.json and replaced before. Real words in capitals (NICHT, ACHTUNG) stay: the model
# reads them correctly after lower-casing.
_LETTER_NAMES = {"A": "ah", "B": "beh", "C": "zeh", "D": "deh", "E": "eh", "F": "eff", "G": "geh", "H": "hah",
                 "I": "ih", "J": "jott", "K": "kah", "L": "ell", "M": "emm", "N": "enn", "O": "oh", "P": "peh",
                 "Q": "kuh", "R": "err", "S": "ess", "T": "teh", "U": "uh", "V": "fau", "W": "weh", "X": "iks",
                 "Y": "üpsilon", "Z": "zett", "Ä": "äh", "Ö": "öh", "Ü": "üh"}
# short German words written in capitals ("DAS IST GUT"): not spelled out
_SHORT_WORDS = set("""AB AN AM AUF AUS BEI BIS DA DAS DEM DEN DER DES DIE DU EIN ER ES FÜR GUT HER HIN ICH IHR IM IN IST JA
JE MAL MIT NEU NIE NUN NUR OB OFT SIE SO TAG UM UND UNS VOM VON VOR WAS WER WIE WIR WO ZU ZUM ZUR ALT NEIN DOCH NOCH
SEHR HIER JETZT BITTE DANKE ALLE ODER ABER AUCH""".split())
# longer abbreviations with a vowel that are spelled out anyway (otherwise: 4+ letters with a vowel = a word)
_ALWAYS_SPELLED = {"ADAC", "DSGVO", "ISBN", "SUV", "DFB", "HTTPS", "VPN", "UKW", "IHK",
                   "AOK", "ABS", "GPS", "USB", "FAQ", "EDV", "OECD", "UNHCR", "WHO", "AGB", "BGB", "STVO",
                   "IOS", "USA", "EU", "ARD", "CDU", "CSU", "SPD", "FDP", "AFD", "ICE", "IC", "DB", "ÖPNV", "HIV"}
# Single letters (formulas, "Plan B") are spoken by their names. Not: hyphenated words (U-Bahn, E-Mail, x-mal) and
# the English "I".
_SINGLE_LETTER = re.compile(r"(?<![\wÀ-ɏ'’.-])([a-zA-ZäöüÄÖÜ])(?![\wÀ-ɏ'’-]|\.[a-zA-Z])")
# lower-case runs without a vowel (paths, short forms: "srv", "mnt")
_NO_VOWEL = re.compile(r"(?<![\wÀ-ɏ'’-])([bcdfghjklmnpqrstvwxzß]{2,5})(?![\wÀ-ɏ'’-])")
_INTERJECTIONS = {"hm", "hmm", "pst", "psst", "brr", "grr", "pff", "pfff", "tss", "sch", "st", "ts", "pscht"}
_UPPERCASE_ABBR = re.compile(r"(?<![\wÀ-ɏ])([A-ZÄÖÜ]{2,6})(s?)(?![\wÀ-ɏ])")
_VOWELS = set("AEIOUÄÖÜY")


def _spelled(word: str, plural: str) -> str:
    return " ".join(_LETTER_NAMES[c] for c in word) + plural


def spell_out_letters(text: str) -> str:
    """Spell out upper-case abbreviations, single letters and vowel-less runs with German letter names."""
    if not text:
        return text
    _abbreviation_rules()                                     # fills _ACRONYM_WORDS

    def shouting(m: re.Match) -> bool:
        """A neighbour word also in capitals ("BITTE SOFORT MELDEN") means these are words - but only if one of
        them is a long word with a vowel. Only direct neighbours without punctuation in between."""
        before = re.search(r"([A-Za-zÄÖÜäöüß]+)[ \t]+$", m.string[:m.start()])
        after = re.match(r"[ \t]+([A-Za-zÄÖÜäöüß]+)", m.string[m.end():])
        neighbours = [x.group(1) for x in (before, after) if x and x.group(1).isupper() and len(x.group(1)) >= 2]
        return any(len(x) >= 4 and _VOWELS & set(x) for x in neighbours + [m.group(1)]) and bool(neighbours)

    def replace(m: re.Match) -> str:
        w, plural = m.group(1), m.group(2)
        if w in _SHORT_WORDS or re.fullmatch(r"[IVX]+", w):        # Roman numerals (Ludwig II) stay
            return m.group(0)
        if w in _ACRONYM_WORDS:                                    # plural of a table word: RAMs -> Ramms
            return _ACRONYM_WORDS[w] + plural
        has_vowel = bool(_VOWELS & set(w))
        if w in _ALWAYS_SPELLED or not has_vowel:
            return _spelled(w, plural)
        if shouting(m):
            return m.group(0)
        if len(w) <= 3:
            return _spelled(w, plural)
        return m.group(0)                                          # 4+ letters with a vowel: a word (NICHT)

    text = _UPPERCASE_ABBR.sub(replace, text)
    text = _NO_VOWEL.sub(lambda m: m.group(0) if m.group(1) in _INTERJECTIONS
                         else " ".join(_LETTER_NAMES[c.upper()] for c in m.group(1) if c.upper() in _LETTER_NAMES), text)
    return _SINGLE_LETTER.sub(lambda m: m.group(0) if m.group(1) == "I" else _LETTER_NAMES[m.group(1).upper()], text)


# ------------------------------------------------------------------------------------------------ markup
#
# Language models answer in Markdown: rules, headings, asterisks, tables, bullets. The model would read them out
# ("Sternchen Sternchen") or stumble over empty pieces.
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_RULE_LINE = re.compile(r"(?m)^[ \t]*([-*_])[ \t]*(?:\1[ \t]*){2,}$")
_HEADING = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*")
# Markdown bullets are only -, * and + (and the bullet sign); a dash at the start of a line is punctuation.
_BULLET = re.compile(r"(?m)^[ \t]{0,6}[-*+\u2022][ \t]+")
_MONTHS = "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"
# "25. Juli 2000" at the start of a line is a date, not a list number. (?![ \t]) holds all the spaces, otherwise the
# month would slip through after backtracking by one space.
_LIST_NUMBER = re.compile(r"(?m)^[ \t]{0,6}\d{1,2}[.)][ \t]+(?![ \t])(?!(?:" + _MONTHS + r")\b)")
_QUOTE = re.compile(r"(?m)^[ \t]{0,3}>[ \t]?")
_TABLE_ROW = re.compile(r"(?m)^[ \t]*\|(.*)\|[ \t]*$")
_TABLE_SEPARATOR = re.compile(r"(?m)^[ \t]*\|[\s:|-]+\|[ \t]*$")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
# An underscore inside a name ("selbst_bauen") becomes a word border; before the emphasis rule, otherwise two such
# names in one sentence would turn into italics.
_NAME_UNDERSCORE = re.compile(r"(?<=\w)_+(?=\w)")
# ".py" -> "Punkt Python". The period must stand right before "py" with no word character after it; a sentence
# period after "daten.py." stays, ".python" and ".pyc" are left alone.
_PY_SUFFIX = re.compile(r"\.py(?!\w)")
_BLANK_LINES = re.compile(r"\n{2,}")
# E-mail and web addresses: the model knows neither "@" nor "/". Endings such as .de are spelled, .com/.org/.dev
# read as a word.
_TLD_NAMES = {"de": "deh eh", "at": "ah teh", "ch": "zeh hah", "eu": "eh uh", "io": "ih oh", "ai": "ah ih",
              "uk": "uh kah", "us": "uh ess", "fr": "eff err", "it": "ih teh", "nl": "enn ell", "tv": "teh fau",
              "me": "emm eh", "com": "komm", "net": "nett", "org": "org", "dev": "dev", "app": "äpp", "info": "info",
              "edu": "eh deh uh", "gov": "gaff", "online": "onlein", "shop": "schopp"}
_EMAIL = re.compile(r"(?<![\w.@])([\w.+-]+)@([\w-]+(?:\.[\w-]+)*\.([a-zA-Z]{2,6}))\b")
_URL = re.compile(r"(?<![\w@/.])(?:https?://)?((?:www\.)?(?:[\w-]+\.)+(" + "|".join(sorted(_TLD_NAMES, key=len, reverse=True))
                  + r"))\b((?:/[\w~%+-]+(?:\.[\w~%+-]+)*)*/?)(?:\?\S*)?")


def _domain(d: str) -> str:
    parts = d.split(".")
    words = ["weh weh weh" if t.lower() == "www" else t for t in parts[:-1]] + [_TLD_NAMES.get(parts[-1].lower(), parts[-1])]
    return " Punkt ".join(w.replace("-", " Strich ") for w in words)


def _email(m: re.Match) -> str:
    who = re.sub(r"[._]", " Punkt ", m.group(1)).replace("-", " Strich ").replace("+", " plus ")
    return f"{who} at {_domain(m.group(2))}"


def _url(m: re.Match) -> str:
    # without scheme and without www/known ending it is no address: "z.B." and "Version 3.2" stay
    path = " Schrägstrich ".join(x for x in m.group(3).split("/") if x)
    return _domain(m.group(1)) + (" Schrägstrich " + path if path else "")


_SYMBOL_LINE = re.compile(r"(?m)^[ \t]*[^\wÀ-ɏ\n]+[ \t]*$")

# Emoji and smileys are removed (as a space, so "super😂toll" does not become "supertoll"): pictographs and
# emoticons, dingbats/symbols, arrows-symbols, flags, and the invisible joiners (variation selector FE0F, joiner 200D,
# keycap 20E3, flag tags).
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF"
                    "\uFE0E\uFE0F\u200D\u20E3\U000E0020-\U000E007F\u3030\u303D]+")
# text smileys only when free-standing: ":)" yes, "12:30" and "1:3" no
_SMILEY = re.compile(r"(?<![\w])(?:[:;=][-'o^]?[)(\]\[DPpOo3|/\\]+|[xX]D+|\^\^|<3"
                     r"|¯\\_\(ツ\)_/¯)(?=\s|$|[.,!?])")
# arrows ("Berlin → Hauptstadt", "A -> B") are a pause, not a word
_ARROW = re.compile("[ \t]*(?:[\u2190-\u21FF\u27F5-\u27FF]|->|=>)[ \t]*")


def _drop(m: re.Match) -> str:
    """Remove an emoji/smiley. If it stood in place of punctuation ("regnen :) Nimm ..."), a comma keeps the
    pause."""
    before = m.string[:m.start()].rstrip()
    after = m.string[m.end():].lstrip()
    if before and before[-1].isalnum() and after[:1].isupper():
        return ", "
    return " "


def _list_word(m: re.Match) -> str:
    """List number as an enumeration word: 1. -> Erstens, 3) -> Drittens (ordinal + "ns")."""
    n = int(re.search(r"\d+", m.group(0)).group(0))
    if n == 0:
        return ""
    return num2words(n, lang="de", to="ordinal").capitalize() + "ns, "


def clean_markup(text: str) -> str:
    """Turn Markdown (and LaTeX, emoji, addresses) into something that can be read aloud."""
    if not text:
        return text
    text = _CODE_FENCE.sub(" ", text)
    text = latex(text)                                       # before emphasis: _ and * belong to the formula there
    text = re.sub(r"(?<=[\w)])\s+\*\s+(?=[\w(])", " mal ", text)   # 5 * 3, before the asterisks go
    text = re.sub(r"(?<=\d)\*(?=\d)", " mal ", text)                 # 5*3
    text = text.replace("`", "")
    text = _EMOJI.sub(_drop, text)
    text = _SMILEY.sub(_drop, text)
    text = _ARROW.sub(", ", text)
    text = _TABLE_SEPARATOR.sub("", text)                    # |---|---| goes entirely
    text = _TABLE_ROW.sub(lambda m: m.group(1).strip().replace("|", ", "), text)
    text = _RULE_LINE.sub("", text)
    text = _HEADING.sub("", text)
    text = _QUOTE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _NAME_UNDERSCORE.sub(" ", text)
    text = _PY_SUFFIX.sub(" Punkt Python", text)
    text = _EMAIL.sub(_email, text)
    text = _URL.sub(_url, text)
    # Emphasis before list numbers: "**3. Wirtschaft:**" starts with asterisks, the number is at the line start only
    # afterwards.
    text = _EMPHASIS.sub(r"\2", text)
    # Asterisks and underscores left over belong to nothing (e.g. bold cut at a chunk border).
    text = text.replace("*", "").replace("_", " ")
    text = _BULLET.sub("", text)
    text = _LIST_NUMBER.sub(_list_word, text)               # "2. " -> "Zweitens, "
    text = _SYMBOL_LINE.sub("", text)                       # lines without any letter
    text = _BLANK_LINES.sub("\n", text)                     # blank lines -> one line break
    # Lines are pauses. A heading ("Wirtschaft:") or list item gets a sentence end, otherwise the model reads it into
    # the next sentence. Line breaks never occurred in training, so they become spaces.
    text = re.sub(r"(?m):[ \t]*$", ".", text)
    text = re.sub(r"(?m)([\wÀ-ɏ)\"“”'])[ \t]*$", r"\1.", text)
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    return text


# ------------------------------------------------------------------------------------------------ mathematics
#
# Formulas come as LaTeX ($a^2 + b^2 = c^2$, \frac, \sqrt) or Unicode (E = mc², √2, π). Two stages: latex() early in
# clean_markup (before emphasis eats "_" and "*"), math_symbols() after the units, so "85 m²" is already
# "Quadratmeter" and not "m hoch zwei".
_GREEK = {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma", "delta": "Delta",
          "epsilon": "Epsilon", "varepsilon": "Epsilon", "zeta": "Zeta", "eta": "Eta",
          "theta": "Theta", "vartheta": "Theta", "iota": "Iota", "kappa": "Kappa",
          "lambda": "Lambda", "mu": "My", "nu": "Ny", "xi": "Xi", "pi": "Pi", "rho": "Rho",
          "sigma": "Sigma", "tau": "Tau", "upsilon": "Ypsilon", "phi": "Phi", "varphi": "Phi",
          "chi": "Chi", "psi": "Psi", "omega": "Omega"}
_LATEX_COMMANDS = {"cdot": " mal ", "times": " mal ", "div": " geteilt durch ", "pm": " plus minus ",
                   "mp": " minus plus ", "approx": " ungefähr ", "neq": " ungleich ", "ne": " ungleich ",
                   "leq": " kleiner gleich ", "le": " kleiner gleich ", "geq": " größer gleich ",
                   "ge": " größer gleich ", "lt": " kleiner als ", "gt": " größer als ",
                   "infty": " unendlich ", "to": " gegen ", "rightarrow": " gegen ",
                   "Rightarrow": ", daraus folgt, ", "implies": ", daraus folgt, ",
                   "iff": " genau dann, wenn ", "in": " Element von ", "sum": " Summe ",
                   "prod": " Produkt ", "int": " Integral ", "lim": " Limes ", "sin": " Sinus ",
                   "cos": " Kosinus ", "tan": " Tangens ", "log": " Logarithmus ",
                   "ln": " natürlicher Logarithmus ", "exp": " e hoch ", "partial": " partiell ",
                   "nabla": " Nabla ", "cdots": " und so weiter ", "ldots": " und so weiter ",
                   "dots": " und so weiter ", "circ": " Grad ", "degree": " Grad ", "percent": " Prozent "}
_LATEX = re.compile(r"\$\$(.+?)\$\$|\$(?=\S)([^$\n]+?)(?<=\S)\$(?!\d)|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)
_ROOT_ORDINALS = {2: "zweite", 3: "dritte", 4: "vierte", 5: "fünfte"}


def _latex_fraction(m: re.Match) -> str:
    num, den = m.group(1).strip(), m.group(2).strip()
    # compound numerator: "..., geteilt durch 2 a" instead of "... durch 2 a"
    compound = bool(re.search(r"\s|[-+=]|\\pm", num))
    return f" ({num}){', geteilt' if compound else ''} durch ({den}) "


def _root(degree: str, content: str) -> str:
    # "³".isdigit() is True but int("³") fails: map superscript digits first
    g = degree.translate(_SUPERSCRIPT).strip()
    word = (_ROOT_ORDINALS.get(int(g), g + '-te') if g.isdecimal() else g)
    return f" {word} Wurzel aus ({content}) "


def _latex_to_text(s: str) -> str:
    # limits belong to the operator: "Integral von 0 bis 1", not "Integral 0 hoch 1"
    s = re.sub(r"\\int\s*_\s*\{?([^{}\s^]+)\}?\s*\^\s*\{?([^{}\s]+)\}?", r" Integral von \1 bis \2 ", s)
    s = re.sub(r"\\(sum|prod)\s*_\s*\{([^{}=]+)=([^{}]+)\}\s*\^\s*\{?([^{}\s]+)\}?",
               lambda m: f" {'Summe' if m.group(1) == 'sum' else 'Produkt'} von {m.group(2)} gleich {m.group(3)} bis {m.group(4)} über ", s)
    s = re.sub(r"\\lim\s*_\s*\{([^{}]+?)\s*\\to\s*([^{}]+)\}", r" Limes für \1 gegen \2 von ", s)
    previous = None
    while previous != s:                 # from the inside out, nested
        previous = s
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", _latex_fraction, s)
        s = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}",
                   lambda m: _root(m.group(1), m.group(2)), s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r" Wurzel aus (\1) ", s)
        s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname|textrm)\s*\{([^{}]*)\}", r" \1 ", s)
        s = re.sub(r"\^\s*\{([^{}]*)\}", r" hoch (\1) ", s)
        s = re.sub(r"_\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\^\s*\\circ", " Grad ", s)
    s = re.sub(r"\\sqrt\s*(\w)", r" Wurzel aus \1 ", s)
    s = re.sub(r"\^\s*(-?\w)", r" hoch \1 ", s)
    s = re.sub(r"_\s*(\w)", r" \1 ", s)
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\b", " ", s)
    s = re.sub(r"\\[,;:! ]|\\q?quad\b", " ", s)
    s = re.sub(r"\\([A-Za-z]+)",
               lambda m: _LATEX_COMMANDS.get(m.group(1)) or (" " + _GREEK[m.group(1).lower()] + " "
                                                             if m.group(1).lower() in _GREEK else " " + m.group(1) + " "), s)
    s = s.replace("{", " ").replace("}", " ")
    s = re.sub(r"([A-Za-z])'", r"\1 Strich ", s)                     # f'(x)
    # f(x) -> f von x, Sinus von x. Other letters before a bracket are rather a product: n(n+1) is n mal (n plus 1)
    s = re.sub(r"\b(Sinus|Kosinus|Tangens|Logarithmus)\s*\(([^()]*)\)", r"\1 von \2", s)
    s = re.sub(r"\b([fghFGHP](?:\s+Strich)*)\s*\(([^()]*)\)", r"\1 von \2", s)
    s = re.sub(r"\b([a-zA-Z0-9])\s*\(", r"\1 mal (", s)
    s = s.replace("(", " ").replace(")", " ")
    s = re.sub(r"(?<=\w)\s*!", " Fakultät", s)
    s = re.sub(r"\s*=\s*", " gleich ", s)
    s = re.sub(r"\s*\+\s*", " plus ", s)
    s = re.sub(r"\s*[-−]\s*", " minus ", s)
    s = re.sub(r"\s*[*·]\s*", " mal ", s)
    s = re.sub(r"\s*/\s*", " durch ", s)
    s = re.sub(r"\s*<\s*", " kleiner als ", s)
    s = re.sub(r"\s*>\s*", " größer als ", s)
    # "2a", "4ac": number and variable spoken apart
    s = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s)
    s = re.sub(r"\b([a-z])([a-z])\b", r"\1 \2", s)      # ac, mc, dx -> a c, m c, d x
    # numbers are written out here already, otherwise expand_units would read "x hoch 2 d x" as "2 d" = two days
    return re.sub(r"\s{2,}", " ", numbers(s)).strip()


def latex(text: str) -> str:
    """Read LaTeX formulas ($…$, $$…$$, \\(…\\), \\[…\\]) as German words."""
    return _LATEX.sub(lambda m: " " + _latex_to_text(next(g for g in m.groups() if g is not None)) + " ", text)


_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_MATH_SYMBOLS = {"√": " Wurzel aus ", "∛": " dritte Wurzel aus ", "π": " Pi ", "∞": " unendlich ",
                 "≠": " ungleich ", "÷": " geteilt durch ", "∑": " Summe ", "∫": " Integral ",
                 "∂": " partiell ", "∆": " Delta ", "Δ": " Delta ", "α": " Alpha ", "β": " Beta ",
                 "γ": " Gamma ", "δ": " Delta ", "ε": " Epsilon ", "θ": " Theta ", "λ": " Lambda ",
                 "σ": " Sigma ", "φ": " Phi ", "ω": " Omega ", "Ω": " Omega ", "ρ": " Rho ",
                 "τ": " Tau ", "η": " Eta ", "∈": " Element von ", "∀": " für alle ",
                 "∃": " es gibt "}


# A dash between two operands. "8.000–12.000", "10-15", "1990 – 2000" are ranges ("bis"); in a calculation it is
# minus: "10 – 4 = 6", "x = 10 - 4", "3 + 5 - 2". A calculation means: in the formula around the dash (numbers,
# single letters, operators and operator words up to the first real word) there is a comparison or another operator
# - the whole sentence would be too coarse: "20–25 °C, also 22 ± 3 Grad" is still a range. Also arithmetic
# questions ("Was ergibt 10 - 4?") and results ("10 - 4 ergibt 6"). The minus sign U+2212 is always minus.
_DASH = re.compile(r"(\b\d+(?:[.,]\d+)*|\b[a-zA-Z]\b|\))([ \t]*)([-–−])([ \t]*)"
                   r"(?=\(|\d|[a-zA-Z](?![\wÀ-ɏ]))")
_FORMULA_PART = (r"(?:\d+(?:[.,]\d+)*|\b[a-zA-Z]\b|\b(?:plus|minus|mal|hoch|geteilt|durch)\b"
                 r"|[=≠≈≤≥±<>+*×·÷^()−–-])")
_FORMULA_LEFT = re.compile(r"(?:" + _FORMULA_PART + r"[ \t]*)+$")
_FORMULA_RIGHT = re.compile(r"(?:[ \t]*" + _FORMULA_PART + r")+")
_OPERATOR = re.compile(r"[=≠≈≤≥±<>+*×·÷^−]|\b(?:plus|mal|hoch|geteilt)\b")
_QUESTION_BEFORE = re.compile(r"\b(?:ergibt|ergeben|rechne|berechne|subtrahiere)[ \t]*$", re.I)
_RESULT_AFTER = re.compile(r"[ \t]*(?:ergibt|ergeben|ist|sind|gleich)[ \t]+(?:minus[ \t]+)?\d")
_SENTENCE_BOUNDARIES = (". ", "! ", "? ", "; ", "\n")
_TIMES_COUNT = re.compile(r"(?<![\w.,])(\d+)(?:[ \t]?×|x)(?=[ \t]+[^\W\d]|[ \t]*[.,;:!?)]|[ \t]*$)")


def _times_count(m: re.Match) -> str:
    """"2x täglich" is zweimal; "3x + 2" and "= 3x" are drei x (a variable)."""
    after = m.string[m.end():m.end() + 12]
    before = m.string[max(0, m.start() - 10):m.start()]
    if re.match(r"[ \t]+(?:plus|minus|mal|gleich|hoch|durch|geteilt|kleiner|größer)\b|[ \t]*[-+=<>*/^]", after) \
            or re.search(r"(?:[=+<>]|\bgleich|\bplus|\bminus)[ \t]*$", before):
        return f"{m.group(1)} x"
    return ("ein" if m.group(1) == "1" else _cardinal(int(m.group(1)))) + "mal"


def _is_calculation(s: str, m: re.Match) -> bool:
    start = max(s.rfind(z, 0, m.start()) for z in _SENTENCE_BOUNDARIES) + 1
    ends = [i for i in (s.find(z, m.end()) for z in _SENTENCE_BOUNDARIES) if i >= 0]
    end = min(ends) if ends else len(s)
    left = _FORMULA_LEFT.search(s, start, m.start())
    l0 = left.start() if left else m.start()
    right = _FORMULA_RIGHT.match(s, m.end(), end)
    r1 = right.end() if right else m.end()
    formula = s[l0:m.start()] + m.group(1) + " " + s[m.end():r1]
    if _OPERATOR.search(formula):
        return True
    if _QUESTION_BEFORE.search(s, start, l0) and not re.search(r"[^\W\d_]", s[r1:end]):
        return True
    return bool(_RESULT_AFTER.match(s, r1, end))


def _dash(m: re.Match) -> str:
    left, sign = m.group(1), m.group(3)
    numeric = left[0].isdigit() and m.string[m.end():m.end() + 1].isdigit()
    if sign == "−" or _is_calculation(m.string, m):
        return f"{left} minus "
    if numeric:
        return f"{left} bis "
    if sign == "-" and m.group(2) and m.group(4):
        return f"{left} minus "
    return m.group(0)


def math_symbols(text: str) -> str:
    """Formulas without LaTeX: a², 10⁻³, H₂O, √2, 2^10, 5 + 3, 10 - 4."""
    if not text:
        return text
    # "mc²" are two quantities, not a word - unless it is a unit (cm²)
    table, _ = _unit_rules()
    text = re.sub(r"\b([a-z]{2})(?=[²³])",
                  lambda m: m.group(1) if (m.group(1) + "²") in table else " ".join(m.group(1)), text)
    text = re.sub(r"(?<=[\w)])([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)",
                  lambda m: " hoch " + m.group(1).translate(_SUPERSCRIPT).replace("-", "minus ") + " ", text)
    text = re.sub(r"[₀₁₂₃₄₅₆₇₈₉]+", lambda m: " " + m.group(0).translate(_SUBSCRIPT) + " ", text)
    text = re.sub(r"(?<=[\w)])\s*\^\s*(-?[\w.,]+)", r" hoch \1", text)
    for symbol, word in _MATH_SYMBOLS.items():
        text = text.replace(symbol, word)
    # operators only between operands: "C++" or a dash in a sentence stay as they are
    text = re.sub(r"(?<=[\w)])\s*\+\s*(?=[\w(])", " plus ", text)
    # "4× so viel", "2x täglich" are multiplicatives, written together ("viermal"); the rest ("4 × 3") is "mal".
    # f(x) also outside LaTeX: "f von x"
    text = re.sub(r"\b([fghFGH])\(([a-zA-Z0-9]{1,3})\)", r"\1 von \2", text)
    # "2x - 1", "3a + b": split number and variable before an operator, then minus/plus apply
    text = re.sub(r"(?<![\w.,])(\d+)([a-zA-Z])(?=[ \t]*[-+=<>*/^−–])", r"\1 \2", text)
    text = _TIMES_COUNT.sub(_times_count, text)
    text = re.sub(r"[ \t]*×[ \t]*", " mal ", text)
    text = re.sub(r"(?<=\d)[ \t]+/[ \t]+(?=\d)", " durch ", text)        # "12 / 4"; "24/7" and "3/4" stay
    text = _DASH.sub(_dash, text)
    text = re.sub(r"(?<=[\w)])\s+·\s+(?=[\w(])", " mal ", text)
    text = re.sub(r"(?<=[\w)])\s+<\s+(?=[\w(])", " kleiner als ", text)
    text = re.sub(r"(?<=[\w)])\s+>\s+(?=[\w(])", " größer als ", text)
    return text


# ------------------------------------------------------------------------------------------------ units
#
# Physical quantities, currencies: "230 V", "5 km/h", "1 h". Only replaced right after a number - "A", "T", "K" or
# "S" are letters in running text, Vitamin C stays Vitamin C. Runs before numbers(): there the 1 would become "eins",
# here it is still known to be "eine Stunde".
_UNIT_RULES: tuple | None = None
_CURRENCY_FIRST = re.compile(r"([$€£¥])\s?(\d+(?:[.,]\d+)*)")


def _unit_rules():
    global _UNIT_RULES
    if _UNIT_RULES is None:
        table = json.loads((DATA / "units.json").read_text(encoding="utf-8")).get("units", {})
        alt = "|".join(re.escape(k) for k in sorted(table, key=len, reverse=True))
        rx = re.compile(r"(?<![\w.,])(-?\d+(?:[.,]\d+)*(?:\s?[-–]\s?\d+(?:[.,]\d+)*)?)"  # number, range
                        r"(\s?)(" + alt + r")(?:/(" + alt + r"))?"                       # unit[/unit]
                        r"(?![\wÀ-ɏ²³])")
        _UNIT_RULES = (table, rx)
    return _UNIT_RULES


# "in einer Stunde", "um einen Grad", "ein Meter": the article depends on the case, which the preposition before
# usually gives away. Two-way prepositions sorted by their most common use with quantities.
_DATIVE = {"in", "mit", "nach", "vor", "seit", "bei", "von", "aus", "zu", "binnen",
           "innerhalb", "ab", "unter", "nach"}
_ACCUSATIVE = {"für", "durch", "gegen", "ohne", "um", "über", "auf"}


def _article(gender: str, before: str) -> str:
    w = re.search(r"(\w+)\s*$", before)
    p = w.group(1).lower() if w else ""
    if p in _DATIVE:
        return "einer" if gender == "f" else "einem"
    if p in _ACCUSATIVE:
        return {"f": "eine", "m": "einen"}.get(gender, "ein")
    return "eine" if gender == "f" else "ein"


def expand_units(text: str) -> str:
    """230 V -> 230 Volt, 1 h -> eine Stunde, 50 km/h -> 50 Kilometer pro Stunde."""
    if not text or not any(c.isdigit() for c in text):
        return text
    table, rx = _unit_rules()
    text = _CURRENCY_FIRST.sub(r"\2 \1", text)            # $5 -> 5 $

    def replace(m: re.Match) -> str:
        number, space, unit, denominator = m.group(1), m.group(2), m.group(3), m.group(4)
        singular, plural, gender, needs_space = table[unit]
        # "16A", "4K-Monitor", "10 T-Shirts": no ampere, kelvin, tesla
        if needs_space and not space:
            return m.group(0)
        if len(unit) == 1 and m.string[m.end():m.end() + 1] == "-":
            return m.group(0)
        # body height and lengths with centimetres: "1,80 m" is "ein Meter achtzig"
        if unit == "m" and not denominator and re.fullmatch(r"\d+,\d\d", number):
            whole, cents = (int(x) for x in number.split(","))
            front = _article("m", m.string[:m.start()]) if whole == 1 else _cardinal(whole)
            return f"{front} Meter" + (f" {_cardinal(cents)}" if cents else "")
        one = number.lstrip("-") == "1"
        word = (singular if one else plural) + (" pro " + table[denominator][0] if denominator else "")
        if one:
            number = ("minus " if number.startswith("-") else "") + _article(gender, m.string[:m.start()])
        return f"{number} {word}"

    return rx.sub(replace, text)


# ------------------------------------------------------------------------------------------------ numbers
#
# Digits are written out in German words (num2words), as all numbers were in the training data.
# after these words "3." is an ordinal - with this ending
_ORDINAL_TEN = {"am", "im", "vom", "zum", "zur", "beim", "dem", "den", "des",
                "einem", "einen", "einer", "jedem", "jeden"}
_ORDINAL_TE = {"der", "die", "das", "ins", "ein", "eine", "jede", "jeder", "jedes"}
# also before punctuation and at the end: "Samstag, den 3., von 16 Uhr", "... den 3."
_ORDINAL = re.compile(r"\b(" + "|".join(sorted(_ORDINAL_TEN | _ORDINAL_TE, key=len, reverse=True))
                      + r")\s+(\d{1,3})\.(?=[\s,;:!?)\]]|$)", re.IGNORECASE)
_DATE = re.compile(r"\b(?:(\d{1,2})\.\s*(bis|und|oder|[–-])\s*)?(\d{1,2})\.\s*(" + _MONTHS + r")\b")
_WEEKDAYS = "Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonntag"
_DATE_PREPOSITIONS = {"bis", "ab", "seit", "vor", "nach"}
# numeric date "29.09.2026", "3.10." -> "29. September 2026", "3. Oktober"; _ORDINAL and _DATE set the ending
# afterwards. Without a year only after a date word ("am", "bis", weekday) - "ab Python 3.12." stays a version.
_MONTH_NAMES = _MONTHS.split("|")
_NUMERIC_DATE = re.compile(r"(?<![\d.,])(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})?(?!\d)")
_DATE_CUE = re.compile(r"\b(?:am|vom|zum|bis|ab|seit|den|dem|der|des|vor|nach|zwischen|" + _WEEKDAYS
                       + r"|Mo|Di|Mi|Do|Fr|Sa|So)\.?,?\s*$", re.IGNORECASE)
# enumeration after a date: "der 14. November, 28.11. und 05.12."
_DATE_LIST = re.compile(r"(?:\b(?:" + _MONTHS + r")\b|\d{1,2}\.\d{1,2}\.)[^.!?\d]{0,40}"
                        r"(?:,|\bund|\boder|\bsowie|\bbis|[–-])\s*$")
# last date in the text before, for the ending in enumerations
_LAST_DATE = re.compile(r"\b(\d{1,2})\.\s*(?:" + _MONTHS + r")\b(?!.*\b\d{1,2}\.\s*(?:" + _MONTHS + r")\b)",
                        re.DOTALL)
_MONTHS_LOWER = {m.lower() for m in _MONTHS.split("|")}
# written-out ordinals 1-31 with their ending, for enumerations
_ORDINAL_WORDS = {num2words(n, lang="de", to="ordinal") + e: e for n in range(1, 32) for e in ("", "n", "r", "m")}
_CLOCK = re.compile(r"\b([01]?\d|2[0-4]):([0-5]\d)\b(?:\s*Uhr\b)?")
_MONEY = re.compile(r"\b(\d+),(\d{2})\s*(€|Euro|US-Dollar|Dollar|Pfund|Franken)\b")
_TIMES = re.compile(r"(?<=\d)\s?[x×]\s?(?=\d)")
_THOUSANDS = re.compile(r"(?<!\d\.)\b\d{1,3}(?:\.\d{3})+\b(?!\.\d)")   # not in IP addresses
_DECIMAL = re.compile(r"\b(\d+),(\d+)\b")
_RANGE = re.compile(r"\b(\d+)\s?[–-]\s?(\d+)\b")
_MINUS = re.compile(r"(?<![\w])-(?=\d)")
_GLUED = re.compile(r"(?<=[A-Za-zÄÖÜäöüß])(?=\d)|(?<=\d)(?=[A-Za-zÄÖÜäöüß])")
_INTEGER = re.compile(r"\b\d+\b")
# Phone numbers digit by digit, groups with a short pause: "+49 170 1234567", "0911 123456", "0911/12345-6".
_PHONE = re.compile(r"(?<![\w.,+])(\+\d{1,3}|0\d{2,5})((?:[ \t]?[/-][ \t]?\d+|[ \t]\d{2,}){1,4})(?![\d.,]\d)")
_PHONE_INTL = re.compile(r"(?<![\w.,])\+(\d{6,15})\b")
# postcode: after "PLZ"/"Postleitzahl" or "in 90402 Nürnberg"
_POSTCODE = re.compile(r"(?:(?<=\bPLZ )|(?<=\bPLZ: )|(?<=\bPLZ ist )|(?<=\bPostleitzahl )|(?<=\bPostleitzahl ist )"
                       r"|(?<=\bin )(?=\d{5}[ \t]+[A-ZÄÖÜ][a-zäöüß]))(\d{5})\b")
# "1.250,50 €": a thousands point before a comma is unambiguous, the point goes
_THOUSANDS_DECIMAL = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{3})+)(?=,\d)")
_NUMBER_SIGN = re.compile(r"#[ \t]?(\d+)\b")


def _digits(s: str) -> str:
    return " ".join(_cardinal(int(c)) for c in s if c.isdigit())


def phone_numbers(text: str) -> str:
    """Phone numbers and postcodes digit by digit. First in the chain: math_symbols would make "+49" "plus 49"."""
    if not text or not any(c.isdigit() for c in text):
        return text
    text = _PHONE.sub(_phone, text)
    text = _PHONE_INTL.sub(lambda m: "plus " + _digits(m.group(1)), text)
    return _POSTCODE.sub(lambda m: _digits(m.group(1)), text)


def _phone(m: re.Match) -> str:
    front = m.group(1)
    parts = [front] + re.findall(r"\d+", m.group(2))
    words = [("plus " + _digits(front[1:])) if front.startswith("+") else _digits(front)] + [_digits(t) for t in parts[1:]]
    return ", ".join(words)


def _cardinal(n: int) -> str:
    return num2words(n, lang="de")


# fractions only if they really are: numerator smaller than denominator, denominator at most 10. "24/7" and "9/11"
# stay pairs of numbers.
_FRACTION = re.compile(r"(?<![\w.,/])(\d{1,2})/(\d{1,2})(?![\w/])")


def _fraction(m: re.Match) -> str:
    num, den = int(m.group(1)), int(m.group(2))
    if not (0 < num < den <= 10):
        return m.group(0)
    name = "halb" if den == 2 else num2words(den, lang="de", to="ordinal").capitalize() + "l"
    return ("ein" if num == 1 else _cardinal(num)) + " " + name


def _ordinal(n: int, ending: str) -> str:
    return num2words(n, lang="de", to="ordinal") + ending          # dritte, dritten, dritter


def _date_ending(before: str) -> str:
    """Ending of the ordinal in a date, read from the word before: "am 25. Juli" -> fünfundzwanzigsten, "der 25. Juli"
    -> fünfundzwanzigste, without article ("Berlin, 25. Juli 2000") -> fünfundzwanzigster."""
    words = re.findall(r"[\wÀ-ɏ]+", before[-80:].lower())
    w = words[-1] if words else ""
    # apposition: "am Freitag, 25. Juli" - the case comes from "am"
    if re.search(r"\b(?:am|vom|zum|seit|bis|ab)\s+(?:" + _WEEKDAYS + r"),\s*$", before, re.I):
        return "n"
    # "am dritten und 4. Mai", "der dritte bis 5. Mai": like the number before
    if w in ("und", "oder", "bis") and len(words) > 1:
        v = words[-2]
        for suffix, e in (("ten", "n"), ("ter", "r"), ("te", "")):
            if v.endswith(suffix) and not v.isdigit():
                return e
    # "der 14. November, 28. November und 5. Dezember": like the date before
    month_idx = -1 if w in _MONTHS_LOWER else (-2 if w in ("und", "oder", "sowie", "bis") and len(words) > 1
                                               and words[-2] in _MONTHS_LOWER else 0)
    if month_idx:
        # already written out ("vierzehnte November, 28. November") or still digits
        if len(words) > -month_idx and words[month_idx - 1] in _ORDINAL_WORDS:
            return _ORDINAL_WORDS[words[month_idx - 1]]
        previous = _LAST_DATE.search(before)
        if previous:
            return _date_ending(before[:previous.start()])
    if w == "zwischen":
        return "m"                                            # zwischen achtundzwanzigstem November
    # "am ersten, 2. und 3. Mai": like the ordinal already written out before
    if w in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[w]
    if w in _ORDINAL_TEN or w in _DATE_PREPOSITIONS:
        return "n"
    if w in _ORDINAL_TE:
        return ""
    return "r"


def _ordinal_after_article(m: re.Match) -> str:
    """"am 3." -> "am dritten". At the end of the text the period was the sentence end as well."""
    ending = "n" if m.group(1).lower() in _ORDINAL_TEN else ""
    return f"{m.group(1)} {_ordinal(int(m.group(2)), ending)}" + ("." if m.end() == len(m.string) else "")


def _numeric_date(m: re.Match) -> str:
    day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return m.group(0)
    # without a year only with a hint at a date: date word before, enumeration after a date, or the two-digit form
    # "28.11." ("Python 3.12." stays)
    before = m.string[:m.start()]
    two_digit = len(m.group(1)) == 2 and len(m.group(2)) == 2
    if not year and not (two_digit or _DATE_CUE.search(before) or _DATE_LIST.search(before)):
        return m.group(0)
    s = f"{day}. {_MONTH_NAMES[month - 1]}" + (f" {year}" if year else "")
    return s + ("." if not year and m.end() == len(m.string) else "")


def _date(m: re.Match) -> str:
    e = _date_ending(m.string[:m.start()])
    day = f"{_ordinal(int(m.group(3)), e)} {m.group(4)}"
    if m.group(1):
        between = "bis" if m.group(2) in ("–", "-") else m.group(2)
        return f"{_ordinal(int(m.group(1)), e)} {between} {day}"
    return day


def _integer(s: str) -> str:
    if (len(s) > 1 and s.startswith("0")) or len(s) > 9:
        return " ".join(_cardinal(int(c)) for c in s)          # phone numbers, identifiers
    n = int(s)
    if len(s) == 4 and 1100 <= n <= 1999:
        return num2words(n, lang="de", to="year")          # neunzehnhundertneunzig
    return _cardinal(n)


def numbers(text: str) -> str:
    """Write out digits in German words. Order matters: first the forms with period, comma and colon, then the bare
    integers."""
    if not text or not any(c.isdigit() for c in text):
        return text
    text = _THOUSANDS_DECIMAL.sub(lambda m: m.group(1).replace(".", ""), text)
    text = _NUMBER_SIGN.sub(r"Nummer \1", text)
    text = _NUMERIC_DATE.sub(_numeric_date, text)
    text = _ORDINAL.sub(_ordinal_after_article, text)
    text = _DATE.sub(_date, text)
    text = _CLOCK.sub(lambda m: ("ein" if int(m.group(1)) == 1 else _cardinal(int(m.group(1))))
                      + " Uhr" + ("" if m.group(2) == "00" else " " + _cardinal(int(m.group(2)))), text)
    # what _CLOCK did not take as a time is a ratio or a score
    text = re.sub(r"(?<![\w.,:])(\d+):(\d+)(?![\d:])", r"\1 zu \2", text)
    text = _MONEY.sub(lambda m: ("ein" if m.group(1) == "1" else _cardinal(int(m.group(1))))
                      + " " + ("Euro" if m.group(3) == "€" else m.group(3))
                      + ("" if m.group(2) == "00" else f" {_cardinal(int(m.group(2)))}"), text)
    text = _TIMES.sub(" mal ", text)                                   # 2 x 3 m
    text = _FRACTION.sub(_fraction, text)                              # 3/4 -> drei Viertel
    text = re.sub(r"(?<![\w.,])1(?=\s+Uhr\b)", "ein", text)              # ein Uhr
    text = re.sub(r"(?<![\w.,])1(?=\s+(?:Million|Milliarde|Billion)\b)", "eine", text)
    text = _THOUSANDS.sub(lambda m: _cardinal(int(m.group(0).replace(".", ""))), text)
    # "Komma," with a short pause: without it the model swallowed it before "fünf"/"vier"
    text = _DECIMAL.sub(lambda m: f"{_integer(m.group(1))} Komma, "
                        + " ".join(_cardinal(int(c)) for c in m.group(2)), text)
    # a period between digits now is a version, IP or decimal: "zwei Punkt fünf"
    text = re.sub(r"(?<=\d)\.(?=\d)", " Punkt ", text)
    text = _RANGE.sub(r"\1 bis \2", text)
    text = _MINUS.sub("minus ", text)
    text = _GLUED.sub(" ", text)
    return _INTEGER.sub(lambda m: _integer(m.group(0)), text)


# ------------------------------------------------------------------------------------------------ the whole chain
def normalize(text: str) -> str:
    """Prepare German text for the model: clean up, then write out units, formulas, numbers and abbreviations, then
    spell out letters - in this order. (The other way round, asterisks around an abbreviation (**z. B.**) would
    keep the abbreviation rule from matching.)"""
    t = spell_out_letters(expand_abbreviations(numbers(math_symbols(expand_units(clean_markup(phone_numbers(text)))))))
    # a slash between words separates them ("Rolls-Royce/Snecma" was read as one word). Only here, after the units:
    # "km/h" is already "Kilometer pro Stunde".
    t = re.sub(r"(?<=[^\W\d_])[ \t]*/[ \t]*(?=[^\W\d_])", " ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return re.sub(r"[ \t]+([,.;:!?])", r"\1", t)   # no space before punctuation


# ------------------------------------------------------------------------------------------------ streaming
# periods after numbers where the stream must not be cut
_NUMBER_PERIODS = [re.compile(r"(?m)^[ \t>]*(?:\*\*|__|\*|_)?\d{1,3}[.)]"),
                   re.compile(r"\d\.(?=[ \t]+\S)")]


class StreamNormalizer:
    """Normalizes a token stream without tearing abbreviations apart at chunk borders.

    :meth:`push` releases only whole sentences (normalized); a short tail is always held back until more text
    arrives or :meth:`flush` is called.
    """

    def __init__(self, normalize_fn=None) -> None:
        """normalize_fn: applied to every released chunk instead of :func:`normalize` (same cut points)."""
        self.rest = ""
        self._normalize = normalize_fn or normalize

    def push(self, text: str) -> str:
        """Add text; return the normalized complete sentences, or "" if none is complete yet.

        Holding back the last characters is not enough: an abbreviation can lie across the cut. So all
        abbreviation spans are marked first, and a sentence end only counts if its period belongs to none of them.
        """
        self.rest += text
        protected = [(m.start(), m.end())
                     for rx, _ in _abbreviation_rules()
                     for m in rx.finditer(self.rest)]
        # a period after a number is mostly no sentence end either: list number ("**3. Wirtschaft**"), ordinal
        # ("am 3. Oktober")
        protected += [(m.start(), m.end()) for rx in _NUMBER_PERIODS
                      for m in rx.finditer(self.rest)]
        # never cut formula blocks ($$ ... $$, \[ ... \]); a block still open protects everything to the end
        for opening, closing in (("$$", "$$"), ("\\[", "\\]")):
            i = 0
            while (a := self.rest.find(opening, i)) >= 0:
                b = self.rest.find(closing, a + len(opening))
                end = len(self.rest) + 1 if b < 0 else b + len(closing)
                protected.append((a, end))
                i = end
        # this close to the end something could still grow
        limit = len(self.rest) - HOLD_BACK
        cut = -1
        for m in re.finditer(r"[.!?\u2026][\s]|\n", self.rest):
            if m.end() > limit:
                break
            if any(a <= m.start() < b for a, b in protected):
                continue          # the period belongs to an abbreviation
            cut = m.end()
        if cut <= 0:
            return ""
        out, self.rest = self.rest[:cut], self.rest[cut:]
        return self._normalize(out)

    def flush(self, text: str = "") -> str:
        """Add the last text and return everything left, normalized."""
        out, self.rest = self.rest + text, ""
        return self._normalize(out)
