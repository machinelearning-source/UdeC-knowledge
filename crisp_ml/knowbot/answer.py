"""Written answer from the fragments ranked first by the model (the last step of KnowBot).

The re-ranker orders the fragments; a language model on Groq then writes the answer using
only the five fragments the model put first, and cites them as [1] to [5]. It needs the
GROQ_API_KEY environment variable; without it the demo shows the ranking only.
"""

import json
import os
import urllib.error
import urllib.request

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS = [os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
          os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")]

INSTRUCTIONS = (
    "You answer questions about the official documents of the University of Cundinamarca. "
    "Use only the numbered fragments given. Cite every fact with the number of its fragment in square brackets, "
    "like [1]. If the fragments do not contain the answer, say so clearly. "
    "Write the whole answer in the language of the question (English or Spanish), "
    "in short paragraphs or a short list."
)


SPANISH_HINTS = ("¿", "á", "é", "í", "ó", "ú", "ñ", " el ", " la ", " los ", " las ", " de ", " para ",
                 " qué ", " cuál ", " cuándo ", " cómo ", " estudiante")


def question_language(question):
    """English or Spanish, from accents and common Spanish words (the two languages KnowBot answers in)."""
    text = f" {question.lower()} "
    return "Spanish" if sum(hint in text for hint in SPANISH_HINTS) >= 2 else "English"


def is_enabled():
    return bool(os.environ.get("GROQ_API_KEY"))


def write_answer(question, fragments, timeout=60):
    """Return (answer, error). fragments: the re-ranked items with document, page and text."""
    if not is_enabled():
        return None, "Written answers are not enabled on this server (GROQ_API_KEY is not set)."
    context = "\n\n".join(
        f"[{number}] {item['document']}, page {item['page']}:\n{item['text']}"
        for number, item in enumerate(fragments, start=1)
    )
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": f"Fragments:\n\n{context}\n\nQuestion: {question}\n\n"
                                    f"Write the whole answer in {question_language(question)}."},
    ]
    last_error = None
    for model in MODELS:
        body = json.dumps({"model": model, "messages": messages, "temperature": 0.1,
                           "max_tokens": 900, "reasoning_effort": "low"}).encode("utf-8")
        request = urllib.request.Request(GROQ_URL, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "User-Agent": "knowbot-reranker/1.0",
        })
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.load(response)
            return data["choices"][0]["message"]["content"].strip(), None
        except (urllib.error.URLError, KeyError, IndexError, ValueError) as error:
            last_error = error
    return None, f"The answer could not be written now ({type(last_error).__name__}). The ranking below is still valid."
