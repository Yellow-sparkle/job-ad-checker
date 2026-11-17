"""Text clean-up that remembers where every character came from.

Before the AI call we remove hidden characters and redact contact details. The AI
then quotes the cleaned text. To show (and score) the quote exactly as it appears
in what the person pasted, every character of the cleaned text keeps the span of
the original text it came from.
"""
import re

ZERO_WIDTH = re.compile("[​-‏⁠-⁤﻿­]")

EMAIL = re.compile(r"([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)")
PROFILE_URL = re.compile(r"((?:https?://)?(?:www\.)?(?:linkedin\.com/in|facebook\.com|instagram\.com|"
                         r"wa\.me|t\.me|x\.com|twitter\.com))/([^\s,;)]+)", re.I)
# 9+ digits, allowing spaces, dots, dashes, brackets and a leading +
PHONE = re.compile(r"(?<![\w£$€])\+?\(?\d[\d\s().-]{7,}\d(?![\w%])")


class MappedText:
    """A string plus, for each character, the (start, end) span in the original."""

    def __init__(self, text, spans=None):
        self.text = text
        self.spans = spans if spans is not None else [(i, i + 1) for i in range(len(text))]

    def sub(self, pattern, repl):
        """Like re.sub. repl(match) returns a list of (new_text, (orig_start, orig_end) | None) parts.
        A part with None keeps the matched characters' own spans (used to keep text as is)."""
        out, spans, pos = [], [], 0
        for m in pattern.finditer(self.text):
            out.append(self.text[pos:m.start()])
            spans.extend(self.spans[pos:m.start()])
            for piece, keep_from in repl(m):
                out.append(piece)
                if keep_from is None:  # whole-match placeholder
                    s = self.spans[m.start()][0] if m.start() < len(self.spans) else 0
                    e = self.spans[m.end() - 1][1] if m.end() > m.start() else s
                    spans.extend([(s, e)] * len(piece))
                else:  # keep the characters of this group with their own spans
                    a, b = keep_from
                    spans.extend(self.spans[a:b])
            pos = m.end()
        out.append(self.text[pos:])
        spans.extend(self.spans[pos:])
        return MappedText("".join(out), spans)


def clean_and_redact(original):
    """Remove hidden characters, then redact emails (keeping the domain), personal
    links and phone numbers. Returns a MappedText over `original`."""
    t = MappedText(original)
    t = t.sub(ZERO_WIDTH, lambda m: [])
    t = t.sub(EMAIL, lambda m: [("[email]", None), ("@" + m.group(2), (m.start(2) - 1, m.end(2)))])
    t = t.sub(PROFILE_URL, lambda m: [(m.group(1), (m.start(1), m.end(1))), ("/[link]", None)])

    def phone(m):
        if len(re.sub(r"\D", "", m.group(0))) < 9:
            return [(m.group(0), (m.start(), m.end()))]
        return [("[phone]", None)]
    t = t.sub(PHONE, phone)
    return t


_QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                              "–": "-", "—": "-", " ": " "})


def _normalised(mt):
    """Lower-case, unify quote marks and dashes, collapse whitespace. Returns
    (normalised text, index into mt.text for each normalised character)."""
    chars, idx = [], []
    prev_space = True
    for i, ch in enumerate(mt.text.translate(_QUOTE_CHARS)):
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        chars.append(ch.lower())
        idx.append(i)
    return "".join(chars), idx


def _norm_quote(q):
    return re.sub(r"\s+", " ", q.translate(_QUOTE_CHARS)).strip().lower()


def find_quote(mt, original, quote):
    """Find `quote` (written by the AI against mt.text) and return the matching
    piece of `original`, or None if it isn't there. Tries the whole quote, then
    the longest part between '...' of at least 3 words."""
    if not quote or not quote.strip():
        return None
    norm, idx = _normalised(mt)
    q = _norm_quote(quote).strip(" .,;:!?\"'")
    candidates = [q]
    parts = [p.strip(" .,;:!?\"'") for p in re.split(r"\.\.\.|…", q)]
    parts = sorted((p for p in parts if len(p.split()) >= 3), key=len, reverse=True)
    candidates += parts
    for c in candidates:
        if len(c) < 4:
            continue
        at = norm.find(c)
        if at < 0:
            continue
        a, b = idx[at], idx[at + len(c) - 1]
        s, e = mt.spans[a][0], mt.spans[b][1]
        return original[s:e]
    return None


_BOUNDARY = re.compile(r"[.!?](?=\s|$)|\n")


def sentence_around(text, start, end):
    """The sentence (or line) of `text` that contains text[start:end]."""
    s = 0
    for m in _BOUNDARY.finditer(text, 0, start):
        s = m.end()
    m = _BOUNDARY.search(text, end)
    e = m.end() if m else len(text)
    return text[s:e].strip()


def mask_phones(text):
    """Hide phone numbers in what we display (quotes keep the original for scoring)."""
    return PHONE.sub(lambda m: "[phone]" if len(re.sub(r"\D", "", m.group(0))) >= 9 else m.group(0), text)
