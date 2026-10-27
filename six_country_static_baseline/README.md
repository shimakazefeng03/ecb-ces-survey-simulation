# Six-Country Static Persona Baseline (c3110)

This is the **baseline** model of the project. An LLM predicts a respondent's answer to ECB Consumer Expectations
Survey (CES) question **c3110** using only static persona characteristics. It gets no macroeconomic context and no
respondent history. Later models (Model 1: + country-level real-time macro context; Model 2: + respondent history)
are scored on the same rows, so any improvement can be measured against this baseline.

| | |
|---|---|
| **Countries** | Belgium, Germany, Spain, France, Italy, the Netherlands (the six original CES countries) |
| **Period** | waves 4–81 (April 2020 to September 2026), data release batch 2026-10-23 |
| **Target** | `c3110`: expected household financial situation 12 months from now compared with today. 1 = get a lot worse, 2 = a little worse, 3 = stay about the same, 4 = a little better, 5 = a lot better |
| **Model run** | Claude Haiku (`claude-haiku-5-5`), effort `low` |
| **Notebook** | `six_country_static_baseline.ipynb` (Google Colab) |

## What was done

1. **Loaded and checked the data.**
   - The static persona file has 1,089,680 respondent-waves from 114,321 respondents.
   - I checked that each (`a0010`, `a0030`) key appears only once, that only the six countries are present, and that
     each wave number agrees with its survey month.
2. **Joined the answers.**
   - The c3110 answer file was joined one-to-one on `a0010` (respondent ID) + `a0030` (wave).
   - All 1,089,680 rows matched, with none left over on either side.
   - Country, date and survey period agreed in both files, with 0 mismatches.
   - Every c3110 value was a valid code from 1 to 5, so no rows were dropped.
3. **Built the persona.**
   - The persona is country plus seven static characteristics: gender, age group, education, household net income
     quintile (within country), household size, number of children aged 0–17 and whether the respondent lives with a partner.
   - The c3110 answer and every other survey answer are kept out of the prompt, and the notebook asserts this.
   - The survey date is also left out. With the date, the model could use what it knows about inflation at that
     time, and that belongs to Model 1.
4. **Drew a fixed evaluation sample.**
   - 200 respondent-waves per country (1,200 rows in total), at most one row per person, seed 42.
   - It is saved as `eval_sample_seed42_n200.csv`.
   - Each row records `n_prior`, the number of earlier c3110 answers by the same person. Results can therefore be
     reported for people with at least 3, 6 and 12 earlier answers, matching the history windows of Model 2.
5. **Ran the LLM.**
   - For each row, Claude Haiku returned a probability for each of the five answers. Its best guess is the most likely option.
   - All 1,200 calls succeeded.
6. **Built four non-LLM comparison baselines.** None of them uses information from the row's own wave or later.
   - **random:** each answer has probability 1/5.
   - **permutation:** the real sample answers, shuffled within each country. The answer mix is correct, but the
     answers carry no information about the individual.
   - **country_mode:** answer shares in the respondent's country in earlier waves.
   - **persona_cell:** answer shares within country × age group × income quintile in earlier waves, falling back to
     the country if the cell has fewer than 30 earlier answers. This is a simple statistical version of a static persona.
7. **Scored everything.**
   - **Accuracy and macro-F1:** higher is better.
   - **Multiclass Brier score:** lower is better. It ranges from 0 to 2, and a uniform guess scores 0.8.
   - **JS divergence** between the predicted and actual answer mix: lower is better.
   - **Entropy ratio and option coverage:** these measure diversity, and 1.0 means the predictions are as spread out as real answers.
   - Each metric is reported overall, by history group and by country.

## Results

### Overall (1,200 rows)

| Method | Accuracy ↑ | Macro-F1 ↑ | Brier ↓ |
|---|---|---|---|
| **LLM (Claude Haiku)** | **0.522** | **0.141** | **0.646** |
| country_mode | 0.521 | 0.138 | 0.642 |
| persona_cell | 0.519 | 0.140 | 0.631 |
| permutation | 0.367 | 0.210 | 1.267 |
| random | 0.193 | 0.149 | 0.800 |

The LLM's entropy ratio was about 0.03 and its option coverage 0.4. In practice it chose the same answer almost every time.

### By history group

These rows are where the history model (Model 2) will be compared. In every group the LLM is about level with
country_mode on accuracy and F1, and it is still behind persona_cell on Brier.

| Group (earlier answers) | Rows | LLM Acc / F1 / Brier | country_mode Acc / F1 / Brier | persona_cell Acc / F1 / Brier |
|---|---|---|---|---|
| all | 1,200 | 0.522 / 0.141 / 0.646 | 0.521 / 0.138 / 0.642 | 0.519 / 0.140 / 0.631 |
| ≥ 3 | 918 | 0.539 / 0.145 / 0.639 | 0.538 / 0.140 / 0.630 | 0.536 / 0.140 / 0.619 |
| ≥ 6 | 730 | 0.538 / 0.147 / 0.641 | 0.536 / 0.140 / 0.633 | 0.534 / 0.139 / 0.624 |
| ≥ 12 | 459 | 0.553 / 0.153 / 0.637 | 0.549 / 0.142 / 0.626 | 0.549 / 0.142 / 0.615 |

### By country

- **Accuracy:** in every country the LLM's accuracy is almost the same as country_mode's. It ranges from about
  0.59 in Germany, the easiest country, to about 0.49 in France, the hardest. The differences between countries
  come from the data: Germans answer "3" most often, while French answers are more spread out.
- **Brier:** persona_cell gives the best Brier score in **all six countries**. The LLM is slightly better than
  country_mode in BE, FR, IT and NL (by 0.003–0.010). It is clearly worse in **Germany (0.617 vs 0.592)** and
  **Spain (0.647 vs 0.630)**, where answers are most concentrated on "3" and the LLM puts too little probability on it.

### Answer distribution

- 52% of real answers are "3, stay about the same", and only 2% are "5, a lot better".
- The LLM's best guess was "3" in **99.3%** of rows and "2" in the other 0.7%. It never picked 1, 4 or 5 as its best guess.
- The probabilities themselves are not all on "3". In the 12-row test, "3" got only about 0.36–0.46, against a real
  share of 0.52, while "1" got about 0.05–0.12 and "4" about 0.11–0.28. Those values were similar across very different profiles.

## Key takeaways

- **A static persona LLM is no better than a country stereotype.** On accuracy, F1 and Brier, Claude Haiku is level
  with country_mode and slightly behind the simple country × age × income model.
- **Accuracy is misleading for this question.** A method that always answers "3" gets about 52% accuracy without
  learning anything, so models should be compared on **Brier** and **macro-F1**.
- **Static traits carry little information about c3110.** People with the same traits answer very differently, and
  without time or economic context the LLM falls back to the typical answer.
- **What later models have to beat:** a Brier score of about **0.63** (persona_cell), a macro-F1 of about **0.14**,
  and better prediction of the less common answers (1, 2, 4, 5).

## Caveats

- Differences below about 0.01 overall, or about 0.02 per country (200 rows each), are within sampling noise. The
  table supports "no better than the baselines", not "worse". No confidence intervals have been computed yet.
- Only one model (Claude Haiku at low effort) has been run. A stronger model might calibrate somewhat better, but it
  is unlikely to change the conclusion.
- The results are unweighted. The survey weight (`ces_weight_raw`) is not used.
- The question and option wording in the prompt should be checked against the official CES Microdata Guide.

## How to run (Colab)

1. Open the notebook in Colab. In **Step 0**, set `DATA_DIR`, the Drive folder that holds the two CSVs, and `OUT_DIR`, where results are saved.
2. Choose the backend:
   - `BACKEND = 'claude'` uses the Anthropic API. Add `ANTHROPIC_API_KEY` in Colab's 🔑 Secrets panel and set
     `CLAUDE_MODEL`, for example `claude-haiku-5-5`.
   - `BACKEND = 'local'` is free. It runs an open model (`Qwen/Qwen2.5-7B-Instruct`, 4-bit) on Colab's free T4 GPU
     (Runtime → Change runtime type → T4 GPU). It takes the model's next-token probabilities for "1" to "5".
3. Run the cells in order. **Step 6b** is a 12-row test, so check it before running **Step 7**, the full 1,200 rows.
   Answers are cached on Drive, so a stopped run picks up where it left off.
4. **Step 10** saves `summary.csv`, `per_country.csv`, `distribution.csv`, `llm_predictions.csv` and `config.json`
   into a time-stamped run folder.

### Data

| File | Contents | Seen by the LLM? |
|---|---|---|
| `six_country_static_persona_baseline_2020_2026.csv` | one row per respondent-wave: country, survey timing, static persona characteristics, survey weight | persona columns only |
| `six_country_c3110_labels_evaluation_only_2020_2026.csv` | the c3110 answer for each respondent-wave | **no** (used only for scoring) |

Both files are derived from the ECB CES microdata. **Do not commit them to this repository.** See `DATA_ACCESS.md`.

## Running (Models 1 and 2)

- Load `eval_sample_seed42_n200.csv` instead of drawing a new sample, so every model is scored on the same 1,200 rows.
- Use the notebook's `score()` function and report results for the same groups: all, ≥3, ≥6 and ≥12 earlier answers.
- Use the same LLM as the baseline, or re-run this baseline with your model. Otherwise the comparison mixes up the
  effect of the model with the effect of the added information.
- History and macro inputs may only use information from waves before the target wave.

## Open items

- Re-run with a second model (for example Sonnet, or the free Qwen backend) as a robustness check.
- Add bootstrap confidence intervals for the differences between methods.
- Confirm what `b2100_prec` measures and whether it belongs in the persona.
- Confirm with the team that c3110 alone is the baseline target. The project plan also mentions c1110 and c1120.
- Extend to all eleven countries once the six-country pipeline is validated.
