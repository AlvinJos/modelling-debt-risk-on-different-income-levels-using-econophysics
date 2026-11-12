# Model review and changelog — 12 Nov 2026

Review of the v1 simulator notebook against the research note ("Modeling Malaysian Debt Risk: A Kinetic Theory Approach to Financial Shock Scenarios"). v2 fixed items 1–6. v3 (the current notebook) adds the features listed under "v3 changes".

## Issues found in v1 (final engine, re-run at 12,000 households)
1. Wages were never credited to households. Only spending and loan payments left their wealth, so about 96% ended with negative wealth.
2. The OPR shock had no effect: final DSR > 60% counts were 538 / 538 / 539 at r = 3% / 4.5% / 8%. The rate only scaled a fixed instalment by (1 + r/52), and no loan balance was tracked.
3. DSR = instalment / income only. The count was fixed at initialisation and only fell as loans matured.
4. The inflation shock ran backwards: the share with negative wealth was 99.3% at 0% inflation and 72.2% at 10%.
5. Loans were assigned the same way in every class (about 68% with loans, median instalment about RM830/month), so "B40 defaults most" was built into the setup.
6. K-S validation was circular: it compared the initial state with itself plus noise.

## Points to check in the research note
- **K-S criterion:** the note says p < 0.05 means the model is valid. p < 0.05 means the distributions differ; p > 0.05 means no evidence of a mismatch. With 100,000+ agents, also report the D statistic, decile shares and the Gini.
- **Eq. 5 (DSR):** adding living expenses into DSR, with expenses at 60% of income, puts every borrower above 0.60 at t = 0. v2 uses BNM's definition (debt payments / income) and a separate liquidity-default rule.
- **SMM:** ε is drawn at random for each trade, so calibrate its distribution or the λ ranges rather than ε itself.
- **Horizon:** the note says 2025–2030 (5 years); v1 ran 10 years. v2 uses 260 weeks.

## v3 changes
1. **Income shocks.** Weekly job-loss hazard by class (placeholders: B40 6%/yr, M40 4%/yr, T20 2.5%/yr), mean spells of 20/16/12 weeks, and 50% of household income lost during a spell. Workers at firms with negative wealth face 3× the hazard. Households start at the steady-state share in a spell.
2. **New borrowing.** Matured loans are renewed and households without a loan take one out, both with placeholder probabilities. Overdrafts are refinanced monthly into a personal loan. All new credit must keep DSR ≤ 60% (`bank_dsr_cap`). Rates were tuned so debt-to-income stays roughly steady over 5 years (B40 +6%, M40 ±1%, T20 −6%).
3. **Starting loans capped at the same DSR limit** (`initial_dsr_cap`, 60%). Without this, high-DSR households only faded out as old loans matured, so year 0 looked artificially risky. This lowers default levels; raise the cap if BNM data show legacy borrowers above 60%.
4. **Default = 13 weeks with the credit line exhausted** (~90 days past due). Defaulted loans and overdrafts are written off. PD is measured only on households not already in default when the scenario starts.
5. **26-week warm-up** under baseline conditions before the scenario clock starts.
6. **Money ledger.** Every flow in or out of the bath is recorded, so conservation is checked to machine precision (relative error ~1e-15).
7. **Automated checks** (`tests/test_model.py`, 10 checks) and **one-at-a-time sensitivity analysis** (±20% on 16 parameters).

## v3 findings worth knowing
- At notebook scale (30,000 households, 10 seeds), the RON95 subsidy removal clearly raises overdraft stress (+0.064 pp, standard error 0.002). Its effect on 5-year default (+0.018 pp, standard error 0.013) is within noise, because most stressed households refinance before defaulting. For small shocks, report overdraft stress alongside PD.
- T20 defaults are rare (~0.1% over 5 years), so T20 PD estimates and sensitivities are noisy at 30,000 households. Use `FULL_SIZE = True` or more seeds for T20-specific claims.
- The sensitivity analysis ranks living-cost shares, income lost during job-loss spells, and the starting DSR cap as the most influential inputs.

## Known limitations
- All loan, spending, buffer and job-loss inputs are placeholders.
- Parameters change abruptly at class thresholds, which creates kinks in the income-decile chart.
- OPR changes pass through to floating loans immediately; real base-rate pass-through has a lag.
- One inflation rate for all classes; B40's basket is more exposed to food and fuel.
- Defaulted households are written off immediately; AKPK restructuring is not modelled.

## Still to do
- Replace placeholder parameters with BNM / AKPK / DOSM / PERKESO data, starting with the most sensitive ones.
- SMM calibration on 2011–2020; forward validation on 2021–2024 (K-S, decile shares, BNM impaired-loan ratios).
- Extensions: pass-through lags, class-specific inflation, AKPK restructuring, urban/rural thresholds.

## Calibration facts used (as of Nov 2026)
- OPR 2.75%, unchanged since July 2025 (held 5 Nov 2026).
- DOSM HIES 2022: B40 ≤ RM5,249; M40 RM5,250–11,819; T20 ≥ RM11,820. Means: B40 RM3,401, M40 RM7,971, T20 RM19,752. Median RM6,338, mean RM8,479.
- RON95 under BUDI95: RM1.99/L since 30 Sep 2025; quota cut to 200 L/month from 1 Apr 2026. Unsubsidised price RM2.60/L (Oct 2025), RM3.87/L (late Mar 2026).
