"""Level 2: answer customer questions grounded in the bill.
Two rounds per bill: (1) the model proposes arithmetic expressions, code evaluates them exactly,
(2) the model writes the final plain-English answers using the verified numbers."""
import json
import logging
import re

from calc import safe_eval
from llm_client import LLMClient, LLMError

log = logging.getLogger("level2")

SYSTEM = """You are a helpful, precise assistant that explains Pakistani electricity bills (K-Electric, LESCO, IESCO) \
to ordinary customers in plain English. You answer using ONLY the information visible on the bill image (and the \
extracted data supplied). You output ONLY a JSON object, no commentary.

ANSWER RULES
- Numbers must be right. Averages, differences, counts, percentages and estimates are checked against the bill. \
Use the code-computed results when given; never do long arithmetic yourself.
- Ground every answer in the bill. Do NOT use outside tariff rates, laws or information. If the bill does not show a \
needed value, say clearly that the bill does not show it instead of guessing.
- Label estimates as estimates and briefly explain how you calculated them (state the numbers used).
- If a question has more than one reasonable meaning, state the meaning you used, or answer both.
- Never reveal personal identifiers (names, addresses, CNIC, account / reference / consumer / meter numbers).
- Write every answer in English even if the bill is partly in Urdu. Amounts as PKR with sensible rounding.
- Net metering / solar bills: usage-history units or billed amounts may be NEGATIVE (more power exported to the grid than imported). \
Never present a negative number as ordinary "usage" or build a normal-sounding average from it. Say that negative values mean net export / credit, \
quote the actual printed numbers, and compare this month's units to the history in that light (e.g. which months were credits and which were not).
- Credits: if an amount is negative or marked CR (arrears, payable amount, previous dues), explain it is a credit in the customer's favour, that nothing \
is due, and that no late-payment surcharge applies when the payable amount is a credit or zero. If the bill says an amount is "NOT TO BE PAID", say so.
- Share / proportion questions ("how much of my bill is taxes / energy / ..."): always state the amount AND the percentage of the current bill \
(percentage computed by code from the printed amounts), plus the printed split if the bill shows one, and list the individual lines.
- Late-payment questions: bills often have several tiers (e.g. "Till 24-Apr" and "After 24-Apr", or 5% and 10% surcharge). Give EVERY tier: \
its date range, the payable amount, and the extra versus paying on time (difference computed by code). Do not describe the highest tier as if it starts on the due date.
- Write with normal spacing: always a space between words and after punctuation, and never join words together.
- Be specific and direct: give the answer first, then 1-3 short sentences of supporting numbers from the bill. \
Keep each answer under ~110 words."""


def _questions_block(questions: list[tuple[str, str]]) -> str:
    return "\n".join(f"{qid}: {q}" for qid, q in questions)


def _plan_prompt(l1: dict, questions) -> str:
    return f"""Customer questions about this bill:
{_questions_block(questions)}

Data already extracted from the bill by an earlier step (may contain mistakes - verify against the image):
{json.dumps(l1)}

Step 1 of 2: decide which questions need arithmetic (averages, differences, percentages, counts over the usage / \
billing history, per-unit costs, estimates, comparisons). For those, write expressions that code will evaluate exactly.
Expressions may use only numbers copied from the bill, + - * / ** ( ), list literals like [141,138,197], and the functions \
avg(list), sum(list), min(list), max(list), round(x, n), abs(x), len(list), count_gt(list, t), count_lt(list, t).
For "how much of X" questions compute the percentage (part / total * 100); for late-payment questions compute each tier's amount minus the on-time amount.
Return JSON: {{"calculations": {{"Q1": {{"short_label": "expression", ...}}, "Q2": {{}} , ...}}}}
Use an empty object for questions that need no arithmetic. Read history tables (13-month usage, billing history) carefully from the image."""


def _answer_prompt(l1: dict, questions, results: dict) -> str:
    return f"""Customer questions about this bill:
{_questions_block(questions)}

Data already extracted from the bill by an earlier step (may contain mistakes - verify against the image):
{json.dumps(l1)}

Exact results computed by code from your expressions (trust these numbers; errors are marked ERROR):
{json.dumps(results, indent=1)}

Step 2 of 2: answer EVERY question following the answer rules. Return JSON mapping each question id to its answer text, e.g. \
{{"Q1": "answer...", "Q2": "answer..."}} with all {len(questions)} ids present."""


def _run_calcs(plan: dict) -> dict:
    results = {}
    calcs = plan.get("calculations", plan) if isinstance(plan, dict) else {}
    if not isinstance(calcs, dict):
        return results
    for qid, exprs in calcs.items():
        if not isinstance(exprs, dict) or not exprs:
            continue
        out = {}
        for label, expr in exprs.items():
            try:
                out[str(label)] = safe_eval(str(expr))
            except (ValueError, TypeError) as e:
                out[str(label)] = f"ERROR ({e}) for: {expr}"
        results[str(qid)] = out
    return results


POLISH_SYSTEM = ("You are a careful proofreader. You output ONLY a JSON object, no commentary.")

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    return sorted(n.replace(",", "").rstrip(".") for n in _NUM_RE.findall(text))


def _fix_spacing(text: str) -> str:
    """Deterministic spacing repair: space after , ; : before a letter, and after a sentence-ending '.' before a capital word."""
    text = re.sub(r"(?<=[,;:])(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[a-z]{2})\.(?=[A-Z][a-z])", ". ", text)
    text = re.sub(r"\b(PKR|Rs\.?)(?=\d)", r"\1 ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def polish_answers(client: LLMClient, bill_id: str, answers: dict[str, str]) -> dict[str, str]:
    """Text-only proofreading pass: fixes missing spaces / typos. A polished answer is accepted only if every
    number in it is unchanged; otherwise the original answer is kept."""
    prompt = ("Fix ONLY spacing (words stuck together, missing space after punctuation), spelling and obvious typos in each answer. "
              "Do not change any number, amount, date, name or meaning, and do not add or remove information.\n"
              f"Return JSON with the same keys: {json.dumps(answers, ensure_ascii=False)}")
    try:
        raw = client.complete_json(POLISH_SYSTEM, prompt)
    except LLMError as e:
        log.warning("%s: polish pass skipped (%s)", bill_id, e)
        return {k: _fix_spacing(v) for k, v in answers.items()}
    out = {}
    for qid, original in answers.items():
        new = str(raw.get(qid, "")).strip()
        ok = new and _numbers(new) == _numbers(original) and 0.6 * len(original) <= len(new) <= 1.4 * len(original)
        out[qid] = _fix_spacing(new if ok else original)
    return out


def answer_questions(client: LLMClient, bill_id: str, image: tuple[str, str], l1: dict,
                     questions: list[tuple[str, str]]) -> dict[str, str]:
    """Returns {question_id: answer}. Always returns a non-empty answer for every question."""
    results: dict = {}
    try:
        plan = client.complete_json(SYSTEM, _plan_prompt(l1, questions), [image])
        results = _run_calcs(plan)
    except LLMError as e:
        log.warning("%s: planning round failed (%s); answering without code calculations", bill_id, e)

    answers: dict[str, str] = {}
    try:
        raw = client.complete_json(SYSTEM, _answer_prompt(l1, questions, results), [image])
        answers = {str(k).strip().upper(): str(v).strip() for k, v in raw.items() if str(v).strip()}
    except LLMError as e:
        log.error("%s: answer round failed: %s", bill_id, e)

    # fill any gap with one single-question retry, then a safe fallback sentence
    for qid, q in questions:
        if answers.get(qid.upper()):
            continue
        try:
            raw = client.complete_json(SYSTEM, _answer_prompt(l1, [(qid, q)], results.get(qid, {}) and {qid: results[qid]}), [image])
            val = next((str(v).strip() for v in raw.values() if str(v).strip()), "")
            if val:
                answers[qid.upper()] = val
        except LLMError as e:
            log.error("%s %s: retry failed: %s", bill_id, qid, e)
        if not answers.get(qid.upper()):
            answers[qid.upper()] = ("I could not read enough of this bill to answer this question reliably, "
                                    "so I am not giving a guess.")
    final = {qid: answers[qid.upper()] for qid, _ in questions}
    return polish_answers(client, bill_id, final)