# Respondent History Windows (Model 2, six countries)

Adds each respondent's own earlier answers to the static persona prompt and varies how many are shown
(window 0 / 3 / 6 / 12). Target, rows, model and scores are those of the
[static baseline](../six_country_static_baseline/), so window 0 is the static baseline re-run under identical conditions.

- Notebook: [`six_country_history_window.ipynb`](six_country_history_window.ipynb)
- Aggregate results: [`results/`](results/)

## Setup

| | |
|---|---|
| Target | c3110, household financial situation in 12 months (1 = a lot worse … 5 = a lot better) |
| Rows | the static baseline's 1,200 evaluation rows (seed 42, 200 per country, ≤ 1 row per person); identity with that run is checked in the notebook |
| History | Yuxin's `six_country_history_up_to12_2020_2026.csv`: the person's most recent N valid c3110 answers from strictly earlier waves, oldest first, shown as "k months earlier: answer" (no calendar dates, so the model gets no macro hint). Joined on a0010 + a0030 and checked against the c3110 answer file on every row |
| Model | `claude-haiku-5-5`, effort low, JSON probabilities over the five options (same request and system prompt as the baseline) |
| Main comparison | the 459 rows with ≥ 12 earlier answers, so every window is full; all models scored on the same rows |
| No-LLM rules | `last_answer` (most recent earlier answer); `history_share_N` (answer shares over the last N earlier answers, prediction = most common, ties to the most recent) |

## Results

Main subset (≥ 12 earlier answers, n = 459):

| Model | Accuracy | Macro-F1 | Brier ↓ | JS div. ↓ | Entropy ratio |
|---|---|---|---|---|---|
| LLM window 0 (static) | 0.553 | 0.150 | 0.639 | 0.511 | 0.02 |
| LLM window 3 | 0.710 | 0.498 | 0.509 | 0.168 | 0.81 |
| **LLM window 6** | **0.739** | **0.643** | 0.473 | 0.130 | 0.86 |
| LLM window 12 | 0.719 | 0.614 | 0.470 | 0.143 | 0.87 |
| persona_cell (no LLM) | 0.549 | 0.142 | 0.615 | 0.523 | 0.00 |
| last_answer (no LLM) | 0.710 | 0.610 | 0.580 | 0.090 | 0.99 |
| history_share_6 (no LLM) | 0.710 | 0.628 | **0.398** | 0.100 | 0.96 |

Paired differences on the same 459 rows (bootstrap 95% CI):

| Comparison | Δ accuracy | Δ Brier |
|---|---|---|
| window 6 vs window 0 | +0.185 [+0.139, +0.231] | −0.166 [−0.187, −0.144] |
| window 6 vs window 3 | +0.028 [+0.000, +0.057] | −0.036 [−0.046, −0.025] |
| window 12 vs window 6 | −0.020 [−0.048, +0.007] | −0.004 [−0.013, +0.006] |
| window 6 vs history_share_6 | +0.028 [+0.000, +0.057] | +0.075 [+0.044, +0.103] |
| window 6 vs last_answer | +0.028 [−0.004, +0.061] | −0.106 [−0.175, −0.040] |

1. **History is the largest improvement over the static persona.** Six earlier answers raise accuracy from 55.3% to
   73.9% and cut the Brier score from 0.639 to 0.473. Every one of the six countries improves.
2. **Six answers are enough.** Going from 3 to 6 earlier answers improves the Brier score, and going from 6 to 12 changes
   neither accuracy nor Brier.
3. **History removes the collapse onto "stay about the same".** The static prompt predicts option 3 for 99.6% of rows.
   With six earlier answers, the predicted answer distribution is close to the real one (JS divergence 0.511 → 0.130,
   entropy ratio 0.02 → 0.86), and all five options are used.
4. **The gain comes mainly from persistence.** 71% of these respondents repeat their previous answer, and the LLM
   predicts the previous answer for 85% of rows. Its accuracy is 2.8 points above the no-LLM rule given the same six answers, but its
   probabilities are less calibrated (Brier 0.473 vs 0.398). On the 133 rows where the answer changed, the LLM gets 27%
   right. Simple answer-share rules on the same history are the bar the LLM models need to clear.
5. **Over all 1,200 rows, history stays useful.** Respondents without history keep the static prompt. Accuracy rises from
   52.2% to 66.8% and Brier falls from 0.649 to 0.524 (window 6).

Window 0 reproduces the static baseline: 52.2% accuracy on all rows, the same as Merry's run, with the same input tokens.
The 136 rows with no earlier answer get the same prompt in every window, and their scores agree across windows
(accuracy 0.471 in all four; Brier 0.673–0.677), so run-to-run noise is small next to the effects above.

Per-country tables, the predicted vs actual distribution, and the stayed/changed split are in [`results/`](results/).

## Next steps for Model 2

- Use window 6 as the default history block.
- Set `MACRO_TEXT` (a function row → text) to Model 1's country-level macro block to run static + macro + history.
- Compare against `history_share_6` as well as the static baseline. Rows where the answer changes are where macro
  context can add value beyond persistence.

## Running it

**Colab:** put the notebook in the team Drive folder next to the static baseline, add `ANTHROPIC_API_KEY` under
🔑 Secrets, then Run all. It reads the same `DATA_DIR` as the static baseline.

**Locally:** put the three Drive files (`six_country_static_persona_baseline_2020_2026.csv`,
`six_country_c3110_labels_evaluation_only_2020_2026.csv`, `six_country_history_up_to12_2020_2026.csv`) into
`history_window/data/`. Put `ANTHROPIC_API_KEY=...` in `history_window/.env`, then run:

```bash
python3 -m venv .venv && .venv/bin/pip install "anthropic>=2.3" "pandas==2.3.3" scipy scikit-learn ipykernel
```

Open the notebook with the `.venv` kernel and run all cells. Answers are cached per window in `data/runs/`, so an
interrupted run resumes where it stopped. A full run is 4,800 calls (3.45M input and 0.23M output tokens, about
$0.46) and takes about 12 minutes.

Respondent-level files (inputs, answer caches, per-row predictions, the executed notebook) stay in `data/`, which is
gitignored. Only aggregate tables go to `results/`.

## Limitations

- The main subset has 459 rows, about 75 per country, so per-country differences are noisy.
- Respondents with ≥ 12 earlier waves are long-time panel members, a selected group. The all-rows results cover the
  full sample.
- This is one model at one effort level, with one call per row and window.
