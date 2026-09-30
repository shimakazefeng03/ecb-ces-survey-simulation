# CES local MPS follow-up

User instruction: execute all six proposed improvement routes locally on PyTorch MPS; do not impose a preset performance gate, monetary/call budget, or token budget.

This is a new experiment tree. Existing raw data, model weights, 660-case results, reports and MLX runs are preserved. Model computations explicitly use MPS with CPU fallback disabled. Classical estimators and data processing use CPU. No paid inference endpoints or rented accelerators are used.

## Required work

1. Extend CES monthly data to all available years, and add country macro factors with documented reference periods, information availability and vintage limitations. Add lagged employment, income, inflation perceptions and growth expectations without current-wave target leakage.
2. Compare direct level prediction, numeric answer revisions, categorical transitions and residual correction under identical input information.
3. Compare recent history, longer-history summary and question-relevant memory with and without macro information on local LLMs.
4. Run heterogeneous local respondent models and role variants; compare against strongest single models and equal-call same-model ensembles; measure error complementarity and calibrate aggregation only on development data.
5. Use deterministic scoring plus local evaluator/optimizer development iterations. Preserve every candidate and diagnostic. Do not edit split membership, outcomes, scorer, or metrics in response to results. No performance threshold is required to continue other experiment branches.
6. Run local LoRA and probability calibration using the improved inputs; retain unadapted comparators, training histories, checkpoint-selection evidence, and repeated-seed results.
7. Freeze finalized configurations and evaluate held-out chronological data. Previously examined 2026 records remain retrospective development/evaluation evidence, not a fresh prospective holdout.

## Evaluation and data limits

Targets retain the six original CES definitions: c1120, c1220, c6120 (continuous); c3010, c3110 (ordered categories); c7010 (binary). Report numerical MAE, categorical accuracy/macro-F1, probability calibration where available, response-change strata, country/month strata, aggregate-distribution fidelity, failures and actual compute usage. Do not replace individual prediction with aggregate accuracy.

Training and validation are chronologically separated. Any learned imputation, standardization, memory selection, residual models, probability temperatures and ensemble weights are fit without evaluation labels. Survey histories strictly precede the target wave. Macro releases must precede the declared information cutoff; latest-vintage lagged data, if used, are labeled retrospective and kept separate from point-in-time verified factors.

The ECB page checked on 2026-09-30 still publishes monthly microdata only through 2026 Q2. A strictly later, never-examined individual-level wave is not currently available. Complete all available experimental work and freeze its configuration; do not claim the prospective evaluation is complete until later microdata can actually be evaluated.

No arbitrary sample cap or token budget is set. Experimental designs, model context capacity, valid answer domains, observed convergence and data availability determine workload. Engineering smoke tests are not empirical results. Record any actual hardware incompatibility rather than silently relabeling MLX as MPS.

Data source: ECB Consumer Expectations Survey. This paper uses data from the ECB Consumer Expectations Survey.
