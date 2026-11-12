# Modelling Debt Risk on Different Income Levels Using Econophysics

Final year project, Department of Physics, Faculty of Science, Universiti Malaya.

## Research question
How does the probability of household debt default differ between Malaysian income groups (B40, M40, T20) over 2025–2030? How sensitive is each group to monetary, fiscal and price shocks?

## Approach
This is an **open-bath bipartite kinetic exchange model**, an extension of the Chakraborti–Chakrabarti saving-propensity model from statistical physics. Households trade with firms the way gas particles exchange energy in collisions, except that each household keeps a fraction λ of its wealth. The system is "open": wages, external demand, loan repayments and profits flow in and out of a reservoir (the "bath") that represents the wider economy. Each scenario is run many times (Monte Carlo) to estimate default probabilities with uncertainty bands.

Shock scenarios, compared against a baseline (OPR 2.75%, CPI 2%, wage growth 3%):

| Scenario | What changes |
|---|---|
| **OPR hike** | Four 25 bp hikes in the first year, from 2.75% to 3.75% |
| **RON95 subsidy removal** | Fuel cost × 2.60/1.99 from month 6, plus one year of extra CPI |
| **Inflation spike** | CPI at 6% for two years |
| **Combined** | All three together |

## Preliminary results
30,000 households, 20 seeds, 5-year horizon:

| Cumulative probability of default | B40 | M40 | T20 |
|---|---|---|---|
| Baseline | 1.1% | 3.4% | 2.4% |
| OPR +100 bp | 1.3% | 3.9% | 2.8% |
| RON95 subsidy removed | 1.3% | 3.9% | 2.6% |
| Inflation spike | 1.5% | 4.2% | 3.0% |
| All three combined | 1.9% | 5.3% | 3.6% |
| *Relative increase, combined vs baseline* | *+71%* | *+54%* | *+46%* |

![Cumulative probability of default by income class](docs/pd_by_class.png)

In this version, M40 has the highest default level and B40 is the most sensitive to shocks. **These numbers depend on placeholder parameters** (see Status) and are not yet thesis results.

## Run it
**Google Colab:** open `notebooks/FYP_debt_risk_simulator_v2.ipynb` in Colab (File → Open notebook → GitHub), then `Runtime → Run all`. It takes about 2 minutes with the default 30,000 households and 20 seeds. Set `FULL_SIZE = True` for 120,000 households.

**Locally:**
```bash
pip install -r requirements.txt
```
```python
import sys; sys.path.insert(0, "src")
from kinetic_debt_model import monte_carlo, SCENARIOS
results = monte_carlo(SCENARIOS, n_seeds=10, weeks=260)
```

## Repository layout
| Path | Contents |
|---|---|
| `notebooks/FYP_debt_risk_simulator_v2.ipynb` | Full pipeline: model, DOSM calibration check, scenarios, Monte Carlo, charts, validation helpers |
| `src/kinetic_debt_model.py` | The model engine on its own (same code as the notebook's first cell) |
| `docs/model-review-2026-11-12.md` | Issues found in v1 and what v2 changed |

## Model in one paragraph
Each week, households receive wages from their employer firm and pay CPI-indexed essential spending (with a fuel component hit by subsidy removal). They service mortgages, hire-purchase and personal loans that carry real balances; floating-rate loans are re-priced at OPR + spread. Households with negative liquid wealth pay card interest. A random 1-to-1 household–firm kinetic exchange (Eq. 1–2 of the research note) redistributes discretionary wealth. Firms take in external demand from the "bath" and pay profit back out to it. A household defaults when its liquid wealth falls below −2 months of income. Income is drawn from a lognormal fitted to DOSM HIES 2022 and matches DOSM's class means and Gini (0.411 simulated vs 0.404).

## Status
⚠️ Saving propensities, loan take-up and sizes, living-cost shares and fuel shares are **placeholders**. They must be replaced with BNM / AKPK / DOSM data before the results are used in the thesis. The last section of the notebook lists each one and where to get the real value.

## Author
Alvin Joseph Albert, Department of Physics, Faculty of Science, Universiti Malaya.

## Key references
- Chakraborti, A., & Chakrabarti, B. K. (2000). Statistical mechanics of money: how saving propensity affects its distribution. *Eur. Phys. J. B*, 17, 167–170.
- Yakovenko, V. M., & Rosser, J. B. (2009). Colloquium: Statistical mechanics of money, wealth, and income. *Rev. Mod. Phys.*, 81, 1703.
- Department of Statistics Malaysia (2023). *Household Income Survey Report 2022*.
- Bank Negara Malaysia. *Financial Stability Review* (household sector).
