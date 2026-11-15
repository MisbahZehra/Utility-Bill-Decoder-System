"""Level 1: extract structured billing data from a bill image with a vision LLM,
then normalise / validate it in code and run one self-check retry."""
import json
import logging
import re
from datetime import datetime

from llm_client import LLMClient, LLMError

log = logging.getLogger("level1")

CHARGE_TYPES = {"energy", "fixed", "fpa", "quarterly_adjustment", "surcharge", "meter_rent", "subsidy", "other"}
TAX_TYPES = {"gst", "electricity_duty", "income_tax", "municipal_tax", "other_tax"}
TOP_KEYS = ["provider", "tariff", "sanctioned_load_kw", "bill_month", "reading_date", "issue_date", "due_date",
            "previous_reading", "current_reading", "units_consumed", "charges", "total_charges", "taxes",
            "total_taxes", "current_bill", "arrears", "payable_within_due_date", "payable_after_due_date"]

SYSTEM = """You are an expert at reading Pakistani electricity bills (K-Electric, LESCO, IESCO) from scanned images, \
screenshots and photos. Bills may be tilted, partly in Urdu, and personal identifiers are covered by grey boxes. \
You output ONLY one JSON object, no commentary."""

PROMPT = """Read the electricity bill image and return ONE JSON object with exactly these keys:

provider (KE | LESCO | IESCO; K-Electric bills are "KE"), tariff, sanctioned_load_kw, bill_month (YYYY-MM), \
reading_date (YYYY-MM-DD), issue_date (YYYY-MM-DD), due_date (YYYY-MM-DD), previous_reading, current_reading, \
units_consumed, charges (list of {"type","amount"}), total_charges, taxes (list of {"type","amount"}), total_taxes, \
current_bill, arrears, payable_within_due_date, payable_after_due_date

RULES
- Copy values AS PRINTED. Never recalculate, round or "fix" amounts. Numbers are JSON numbers in PKR (no commas, no "Rs.").
- Credits/subsidies are NEGATIVE (e.g. "38209CR" -> -38209). A negative previous-dues / credit balance stays negative.
- Dates: "22nd Sep 2026" -> 2026-09-22; month "MAR 2026" / "Mar-26" -> 2026-03. The due date is the last date to pay WITHOUT late surcharge.
- Use null for anything not printed, blank, covered by a grey box or unreadable. Do NOT guess. Use [] for an empty list.
- NEVER output names, addresses, CNIC, account/reference/consumer/contract/meter numbers.
- tariff: code exactly as printed (e.g. "A1-R", "A-1a(01)"). sanctioned_load_kw: value labelled Sanc Load / S.Load / San Load.
- units_consumed: units billed for the current month as printed. previous_reading / current_reading: printed meter readings.
- charges: ONE entry per line of the charges section (skip blank lines and lines printed as 0). Keep repeated types as separate entries (e.g. two energy slab lines). Types:
  energy = cost of units (Variable Upto ... Units, Cost of Electricity, Energy Charges), fixed = fixed charges,
  fpa = fuel price adjustment (use for both FPA and FCA), quarterly_adjustment = QTA / Uniform Quarterly Adjustment,
  surcharge = surcharges (Additional Surcharge (PHL), F.C Surcharge), meter_rent = meter rent / service rent,
  subsidy = government subsidy / relief lines (NEGATIVE amount), other = any other charge line.
- total_charges: the printed charges subtotal (null if none printed).
- taxes: ONE entry per line of the tax / government section. Classify by SECTION, not name. Types:
  gst = sales tax / GST incl. GST on FPA, electricity_duty = electricity duty (ED), income_tax,
  municipal_tax = municipal/local charges under taxes (e.g. KE MUCT (KMC)), other_tax = extra tax, further tax, TV fee etc.
  If only ONE combined tax amount is printed, return taxes = [] and put it in total_taxes.
- total_taxes: printed tax subtotal (null if none printed).
- current_bill: printed amount for THIS month's bill. arrears: previous dues carried forward (keep printed sign, null if blank).
- payable_within_due_date: amount payable by the due date. payable_after_due_date: amount after the due date; if several late amounts are shown, use the HIGHEST.

LAYOUT NOTES
- LESCO / IESCO bills have NO itemised charge lines, only a "Bill Charges Breakdown" summary: Total Electricity Charges, Subsidies, \
Net Electricity Charges, Taxes, Current Bill, Arrears, Installment, Adjustments, Total FPA, Grand Total. For these bills:
  charges = [{"type":"energy","amount":<Total Electricity Charges>}] plus {"type":"subsidy","amount":-<Subsidies>} ONLY if Subsidies is printed, \
plus {"type":"fpa","amount":<Total FPA>} ONLY if a Total FPA amount is printed; total_charges = Net Electricity Charges;
  taxes = [] and total_taxes = the single "Taxes" amount (it is a combined amount); current_bill = "Current Bill";
  arrears = the "Arrears" amount; payable_within_due_date = "Payable within due date" (or Grand Total if that is what the box shows).
- A printed 0 is the value 0 (e.g. arrears 0). Only a BLANK / hidden / unreadable cell is null. A blank Subsidies line adds no charge entry.
- Amounts followed by "CR" are credits and negative (e.g. "54770CR" -> -54770). If an amount has a stray suffix like "39411 / 2", use the number 39411.
- If the payable-after-due-date box says something like "NOT TO BE PAID" or has no amount, use null.
- LESCO/IESCO show two late-payment amounts (e.g. "Upto 14/09/26" and "After 14/09/26", or "Till 12-Oct-26" and "After 12-Oct-26"): use the HIGHEST.
- Do not confuse "LP Surcharge" boxes (just the surcharge amounts) with the payable-after-due-date amounts.
- Several meter-register rows (e.g. off-peak / peak / import / MF): previous_reading and current_reading come from the FIRST register row; \
units_consumed is the printed total units for the month (for K-Electric the big "N units" box).
- Photos may be tilted or blurry; read carefully. Month words: JUL 2026 -> 2026-07, SEP 26 -> 2026-09.

EXAMPLE of a correct output for a K-Electric bill (different bills will have different values):
{"provider":"KE","tariff":"A1-R","sanctioned_load_kw":3,"bill_month":"2026-04","reading_date":"2026-04-03","issue_date":"2026-04-07","due_date":"2026-04-21","previous_reading":8819,"current_reading":8970,"units_consumed":151,"charges":[{"type":"fixed","amount":900},{"type":"energy","amount":1054},{"type":"energy","amount":663.51},{"type":"quarterly_adjustment","amount":52.91},{"type":"fpa","amount":125.27},{"type":"surcharge","amount":64.93}],"total_charges":2860.62,"taxes":[{"type":"electricity_duty","amount":29.41},{"type":"gst","amount":520.21},{"type":"municipal_tax","amount":20}],"total_taxes":569.62,"current_bill":3430.24,"arrears":-0.53,"payable_within_due_date":3430,"payable_after_due_date":3716}

Look carefully at every line item of the bill calculation / charges table, then return the JSON object."""


# ---------------- normalisation helpers ----------------
def to_number(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        n = float(v)
    else:
        s = str(v).strip()
        if not s or s.lower() in {"null", "none", "n/a", "na", "-", "--"}:
            return None
        neg = bool(re.search(r"\bCR\s*$|\d\s*CR\s*$", s, re.I)) or s.startswith("-") \
            or (s.startswith("(") and s.endswith(")")) or s.endswith("-")
        m = re.search(r"\d+(?:\.\d+)?", s.replace(",", ""))
        if not m:
            return None
        n = float(m.group())
        if neg:
            n = -abs(n)
    n = round(n, 2)
    return int(n) if n == int(n) else n


_DATE_FORMATS = ["%Y-%m-%d", "%d-%b-%Y", "%d %b %Y", "%d-%b-%y", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
                 "%d %B %Y", "%d/%b/%Y", "%d-%B-%Y", "%d/%m/%y", "%d %b %y", "%d %B %y", "%d-%b-%y"]


def to_date(v):
    if v is None:
        return None
    s = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", str(v).strip(), flags=re.I).strip()
    for f in _DATE_FORMATS:
        try:
            return datetime.strptime(s, f).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def to_month(v):
    if v is None:
        return None
    s = str(v).strip()
    if re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", s):
        return s
    d = to_date(s)
    if d:
        return d[:7]
    for f in ["%b-%y", "%b %Y", "%B %Y", "%b-%Y", "%B-%Y", "%b %y", "%m/%Y", "%m-%Y", "%Y/%m"]:
        try:
            return datetime.strptime(s, f).strftime("%Y-%m")
        except ValueError:
            continue
    return None


def to_provider(v):
    s = re.sub(r"[^A-Z]", "", str(v or "").upper())
    if s in {"KE", "KESC", "KELECTRIC"} or s.startswith("KELECTRIC"):
        return "KE"
    if s.startswith("LESCO"):
        return "LESCO"
    if s.startswith("IESCO"):
        return "IESCO"
    return None


def _items(raw, allowed, fallback, drop_zero=True):
    out = []
    for it in raw if isinstance(raw, list) else []:
        if not isinstance(it, dict):
            continue
        amt = to_number(it.get("amount"))
        if amt is None or (drop_zero and amt == 0):
            continue
        t = str(it.get("type", "")).strip().lower()
        out.append({"type": t if t in allowed else fallback, "amount": amt})
    return out


def normalize(raw: dict) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    tariff = raw.get("tariff")
    tariff = str(tariff).strip() if tariff not in (None, "") else None
    res = {
        "provider": to_provider(raw.get("provider")),
        "tariff": tariff,
        "sanctioned_load_kw": to_number(raw.get("sanctioned_load_kw")),
        "bill_month": to_month(raw.get("bill_month")),
        "reading_date": to_date(raw.get("reading_date")),
        "issue_date": to_date(raw.get("issue_date")),
        "due_date": to_date(raw.get("due_date")),
        "previous_reading": to_number(raw.get("previous_reading")),
        "current_reading": to_number(raw.get("current_reading")),
        "units_consumed": to_number(raw.get("units_consumed")),
        "charges": _items(raw.get("charges"), CHARGE_TYPES, "other"),
        "total_charges": to_number(raw.get("total_charges")),
        "taxes": _items(raw.get("taxes"), TAX_TYPES, "other_tax"),
        "total_taxes": to_number(raw.get("total_taxes")),
        "current_bill": to_number(raw.get("current_bill")),
        "arrears": to_number(raw.get("arrears")),
        "payable_within_due_date": to_number(raw.get("payable_within_due_date")),
        "payable_after_due_date": to_number(raw.get("payable_after_due_date")),
    }
    return {k: res[k] for k in TOP_KEYS}


def empty_result() -> dict:
    return normalize({})


def _sum_gap(res: dict) -> float:
    """How far the line items are from the printed subtotals (a separately listed FPA line may legitimately sit outside the subtotal)."""
    gap = 0.0
    for items, total in (("charges", "total_charges"), ("taxes", "total_taxes")):
        if res[total] is not None and res[items]:
            full = abs(sum(i["amount"] for i in res[items]) - res[total])
            no_fpa = abs(sum(i["amount"] for i in res[items] if i["type"] != "fpa") - res[total])
            gap += min(full, no_fpa)
    return gap


def extract_bill(client: LLMClient, bill_id: str, image: tuple[str, str]) -> dict:
    """Extract + normalise; one self-check retry when line items don't add up to printed subtotals."""
    best = None
    try:
        best = normalize(client.complete_json(SYSTEM, PROMPT, [image]))
    except LLMError as e:
        log.warning("%s: first extraction failed (%s); retrying", bill_id, e)
        try:
            best = normalize(client.complete_json(SYSTEM, PROMPT, [image]))
        except LLMError as e2:
            log.error("%s: extraction failed: %s", bill_id, e2)
            return empty_result()

    gap = _sum_gap(best)
    if gap > 1.5:
        log.info("%s: line items differ from printed subtotals by %.2f - verifying", bill_id, gap)
        sc = sum(i["amount"] for i in best["charges"])
        st = sum(i["amount"] for i in best["taxes"])
        feedback = (f"{PROMPT}\n\nYour previous answer was:\n{json.dumps(best)}\n\nCHECK FAILED: the charge lines add up to {sc:.2f} "
                    f"(printed total_charges={best['total_charges']}) and the tax lines add up to {st:.2f} "
                    f"(printed total_taxes={best['total_taxes']}). You may have missed, duplicated or misread a line, or "
                    f"put a line in the wrong section. Re-read the bill carefully and return the corrected full JSON. "
                    f"If the printed subtotal itself is what you misread, fix that instead.")
        try:
            second = normalize(client.complete_json(SYSTEM, feedback, [image]))
            if _sum_gap(second) < gap:
                # take the re-read line items/subtotals; keep everything else from the first pass
                # (only filling fields the first pass left empty)
                merged = dict(best)
                for k in ("charges", "total_charges", "taxes", "total_taxes"):
                    merged[k] = second[k]
                for k in TOP_KEYS:
                    if merged[k] is None and second[k] is not None:
                        merged[k] = second[k]
                best = merged
        except LLMError as e:
            log.warning("%s: verification pass failed: %s", bill_id, e)
    return best