# Utility Bill Decoder (WTQ Build Track)

Reads K-Electric / LESCO / IESCO bill images and
- **Level 1**: extracts a strict JSON object per bill (`output/level1.csv`)
- **Level 2**: answers 12 customer questions per bill in plain English (`output/level2.csv`)

**Runtime:** Python 3.12 (3.11+ works)

## Setup
```bash
cd source
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # Windows: copy .env.example .env
# edit .env: set API_KEY (and MODEL_PROVIDER / MODEL_NAME if you change model)
```

## Run on a folder of bills (generates both CSVs)
Expected layout: `test/bills/*.png|jpg`, `test/csv/level1.csv`, `test/csv/level2.csv` (the supplied templates).
```bash
python main.py run --bills ../test/bills --level1 ../test/csv/level1.csv --level2 ../test/csv/level2.csv --out ../output
```
Results are cached per bill in `.cache/`, so an interrupted run resumes; delete `.cache/` to redo everything.
Use `--limit 2` for a quick trial.

## Demo on a single bill
```bash
python main.py demo --bill ../train/bills/KESC_0008.png --ask "How much of my bill is taxes?"
```

## Build the final ZIP
```bash
python package.py --out ../output --zip ../your-submission.zip --demo ../demo/demo.mp4
```
It checks 10 / 120 rows, empty cells and valid JSON, excludes `.env`/caches, and checks the 15 MB limit.

## How it works
1. `imaging.py` fixes orientation, downsizes huge photos, encodes JPEG.
2. `level1.py` sends the image with a rule-heavy prompt (one worked example) to the vision LLM, then **normalises in code**
   (numbers, `CR` -> negative, dates, enum types, drops zero lines, drops any non-schema/identifier keys). If line items do not add up to
   the printed subtotals, it runs **one verification pass** and keeps the re-read items only if they match better.
3. `level2.py` answers in two rounds: the model proposes arithmetic expressions, `calc.py` (a safe AST evaluator) computes them exactly,
   then the model writes grounded English answers using those verified numbers. Estimates are labelled; missing data is stated.
4. `main.py` fills the templates with the `csv` module (column names/order preserved, UTF-8, JSON on one line). Failures on one bill never stop the run
   (null-filled JSON / explicit "could not read" answers).

## Configuration (`.env`)
`MODEL_PROVIDER`, `MODEL_NAME`, `API_KEY`, `MIN_CALL_INTERVAL`. Only the competition's allowed models are accepted (enforced in `config.py`).

## AI usage
- **Models used by the solution:** the single model set in `.env` (default `gemini-3.5-flash-lite` via Google AI Studio; may be switched to `claude-haiku-4-5`, `gpt-4.1-mini` or `gpt-5-mini`). EDIT THIS LINE to match the model you actually ran.
- **APIs/services:** the chosen provider's public REST API only. No OCR libraries, no invoice-extraction services.
- **AI tools used to write the code:** Claude (Anthropic chat assistant) helped write and review the code.
