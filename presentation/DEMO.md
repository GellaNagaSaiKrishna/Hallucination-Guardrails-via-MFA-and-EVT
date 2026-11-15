# Live demo guide

Everything runs locally on your laptop. No internet is needed once the models are cached; they already are.
All commands are run from the project folder: `D:\GNSK-2\SEM-5\Data Science\project`.

## Before the talk (5 minutes)

1. Open a terminal in the project folder, zoom the font to about 150% so the back row can read it, and keep it next to the slides.
2. Do a warm-up run so the models are in the disk cache (the first run is slower):
   ```
   python presentation/demo.py
   ```
3. Close heavy apps (extra browser tabs) to free RAM. The `--rag` step loads a 1.5B-parameter LLM, which needs about 3 GB of free memory.
4. Keep the laptop plugged in.

## During the talk (slide 14, about 3 minutes)

**Step 1: guardrail scores (instant, about 10 s including model loading)**
```
python presentation/demo.py
```
Expected output:

| Question | Type | D² | S(q) | α = 0.01 | α = 0.1 |
|---|---|---|---|---|---|
| 3M FY2018 capital expenditure | ID | 446 | 1.0000 | pass | pass |
| AMD operating margin FY22 | Near-OOD | 655 | 0.027 | pass | **REFUSE** |
| Microsoft lunar data centers | Adversarial | 454 | 1.0000 | pass | pass |
| Who painted the Mona Lisa? | Far-OOD | 891 | 0.0001 | **REFUSE** | **REFUSE** |

What to point out:
- The 3M question is close to the knowledge base (D² 446 < τ₀ 483), so S(q) = 1 and it passes.
- AMD is not indexed. Its distance is in the tail, so it's refused at α = 0.1 but not at 0.01, which shows α working as a safety dial.
- The lunar-data-centre question passes. That's the false-premise limitation: it looks like a normal Microsoft question.
- The Mona Lisa is far away (891) and refused at both settings.
- Each check takes about 1 ms. The first one is a little slower (warm-up).

**Step 2: a full guarded answer (about 30 s; talk while it loads)**
```
python presentation/demo.py --rag "Has CVS Health paid dividends to common shareholders in Q2 of FY2022?"
```
Expected: `blocked = False`, answer *"Yes, CVS Health paid a quarterly cash dividend of $0.55 per share during Q2 of FY2022."*, followed by the 10-K passages it used.

While the LLM loads, say: *"This is the full pipeline: guardrail, then FAISS retrieval, then a local Qwen 1.5B model on the CPU."*

```
python presentation/demo.py --rag "Who painted the Mona Lisa?"
```
Expected: `blocked = True` and the fixed refusal. The LLM is never called, so it is instant after the models load.

**Step 3 (optional): an audience question**
```
python presentation/demo.py -i
```
Type the question, press Enter, and an empty line quits. Good audience prompts: *"Ask something about Pepsi or Walmart"* (indexed, so it should pass) versus *"about Tesla's stock price tomorrow"* or *"a sports question"*.

Avoid promising a specific outcome for audience questions. Short, generic questions sometimes land closer to the corpus than you'd expect.

## If something goes wrong

| Problem | Fallback |
|---|---|
| A model fails to load or something crashes | `python presentation/demo.py --offline` prints the saved results instantly |
| `--rag` says MemoryError | Close apps and retry, or skip step 2 and describe it from slide 11 |
| No time for the demo | Show slide 14; it already lists the four questions and outcomes |

## Other demonstration options

- **Notebook demo:** `notebooks/demo.ipynb` runs the same checks and plots where a question lands relative to the MFA clusters (PCA view). It's good for a visual audience, but needs Jupyter open.
- **Show the report figures:** the score-distribution plot (slide 8) is the best single visual for explaining why some questions pass and others don't.
