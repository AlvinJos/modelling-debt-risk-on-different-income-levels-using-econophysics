# Model review — 12 Nov 2026

Review of the v1 simulator notebook against the research note ("Modeling Malaysian Debt Risk: A Kinetic Theory Approach to Financial Shock Scenarios"). The v2 notebook in `notebooks/` fixes items 1–6.

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

## Still to do
- Replace placeholder parameters with BNM / AKPK / DOSM data (see the table at the end of the notebook).
- SMM calibration on 2011–2020; forward validation on 2021–2024.
- Extensions: new loan origination, unemployment when firms fail, AKPK restructuring, urban/rural thresholds.

## Calibration facts used (as of Nov 2026)
- OPR 2.75%, unchanged since July 2025 (held 5 Nov 2026).
- DOSM HIES 2022: B40 ≤ RM5,249; M40 RM5,250–11,819; T20 ≥ RM11,820. Means: B40 RM3,401, M40 RM7,971, T20 RM19,752. Median RM6,338, mean RM8,479.
- RON95 under BUDI95: RM1.99/L since 30 Sep 2025; quota cut to 200 L/month from 1 Apr 2026. Unsubsidised price RM2.60/L (Oct 2025), RM3.87/L (late Mar 2026).
