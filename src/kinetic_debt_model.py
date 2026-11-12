import numpy as np

# =============================================================================
# FYP: Modelling Debt Risk on Different Income Levels Using Econophysics
# Open-Bath Bipartite Kinetic Exchange Model -- v2 (fixed engine)
#
# Changes from v1 (cell 5 of the original notebook):
#   1. Households now RECEIVE wages every week (paid by their employer firm).
#   2. Loans carry an outstanding balance; floating-rate loans are re-priced
#      from (OPR + spread) every week, so an OPR shock changes instalments.
#   3. Loan take-up and loan size depend on income class.
#   4. Income class is assigned from DOSM HIES 2022 thresholds, not by index.
#   5. Default = absorbing event when liquid wealth falls below a credit limit;
#      DSR (debt payments / income, excl. living costs) tracked separately.
#   6. Scenarios are run over many seeds with common random numbers.
# =============================================================================

WEEKS_PER_MONTH = 52 / 12

# DOSM HIES 2022 (gross monthly household income, RM)
DOSM_2022 = dict(
    median=6338, mean=8479,
    b40_upper=5249, t20_lower=11820,
    class_mean={"B40": 3401, "M40": 7971, "T20": 19752},
    income_share={"B40": 0.161, "M40": 0.376, "T20": 0.463},
)

# Loan products: (name, floating?, spread over OPR or fixed rate, term range in years)
LOAN_TYPES = ["mortgage", "hire_purchase", "personal"]

DEFAULT_PARAMS = dict(
    # --- Income distribution (lognormal fitted to DOSM median & mean) ---
    income_mu=np.log(DOSM_2022["median"]),
    income_sigma=np.sqrt(2 * np.log(DOSM_2022["mean"] / DOSM_2022["median"])),

    # --- Saving propensity lambda by class (uniform ranges; calibrate via SMM) ---
    lambda_range={"B40": (0.01, 0.10), "M40": (0.15, 0.40), "T20": (0.50, 0.85)},
    lambda_firm=0.95,             # firm saving propensity (0 recovers PDF Eq.1-2 exactly)

    # --- Living costs (share of income at t=0, then indexed to CPI) ---
    essential_share={"B40": 0.75, "M40": 0.62, "T20": 0.45},
    fuel_share={"B40": 0.10, "M40": 0.08, "T20": 0.05},   # fuel's share of essential spend

    # --- Initial liquid buffer, in months of income ---
    buffer_months={"B40": (0.0, 2.0), "M40": (1.0, 4.0), "T20": (3.0, 12.0)},

    # --- Loans: take-up probability by class ---
    loan_prob={
        "mortgage":      {"B40": 0.20, "M40": 0.40, "T20": 0.55},
        "hire_purchase": {"B40": 0.35, "M40": 0.45, "T20": 0.45},
        "personal":      {"B40": 0.30, "M40": 0.20, "T20": 0.10},
    },
    # principal as a multiple of ANNUAL income (uniform range)
    loan_size={"mortgage": (2.5, 4.5), "hire_purchase": (0.4, 1.0), "personal": (0.2, 0.6)},
    # remaining term in years (uniform range)
    loan_term={"mortgage": (5, 30), "hire_purchase": (2, 9), "personal": (2, 10)},
    # pricing: floating loans = OPR + spread; HP is fixed-rate in Malaysia
    loan_floating={"mortgage": True, "hire_purchase": False, "personal": True},
    loan_spread={"mortgage": 1.50, "hire_purchase": 3.00, "personal": 5.00},   # % p.a.

    # --- Unsecured overdraft / credit-card debt when liquid wealth < 0 ---
    card_rate=15.0,               # % p.a., not OPR-linked
    credit_limit_months=2.0,      # default if liquid wealth < -limit * monthly income

    # --- Firms and the open bath ---
    firm_capital=(50_000, 10_000),
    external_demand_share=0.15,   # exports/government demand, as a share of the wage bill
    bank_recycle_share=1.0,       # share of loan repayments re-lent to firms
    firm_payout_rate=0.05,        # weekly share of firm wealth above starting capital paid out
                                  # to the bath (taxes, dividends, imports)

    dsr_threshold=0.60,
)


def scenario_path(name, weeks):
    """Return weekly paths: opr (% p.a.), cpi (annual %), wage growth (annual %), fuel multiplier."""
    t = np.arange(weeks)
    opr = np.full(weeks, 2.75)
    cpi = np.full(weeks, 2.0)
    wage_g = np.full(weeks, 3.0)
    fuel = np.ones(weeks)

    if name in ("opr_hike", "combined"):
        for wk in (13, 26, 39, 52):          # four 25bp hikes -> 3.75%
            opr[t >= wk] += 0.25
    if name in ("subsidy_removal", "combined"):
        fuel[t >= 26] = 2.60 / 1.99           # unsubsidised / BUDI95 RON95 price
        cpi[(t >= 26) & (t < 78)] += 1.0      # one-year pass-through to CPI
    if name in ("inflation_spike", "combined"):
        cpi[(t >= 26) & (t < 130)] = np.maximum(cpi[(t >= 26) & (t < 130)], 6.0)
    return dict(opr=opr, cpi=cpi, wage_g=wage_g, fuel=fuel)


SCENARIOS = ["baseline", "opr_hike", "subsidy_removal", "inflation_spike", "combined"]


def annuity_payment(balance, annual_rate_pct, n_weeks):
    r = annual_rate_pct / 100 / 52
    n = np.maximum(n_weeks, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pay = np.where(r > 0, balance * r / (1 - (1 + r) ** (-n)), balance / n)
    return np.where(n_weeks > 0, pay, 0.0)


class KineticDebtModel:
    def __init__(self, N_h=120_000, N_f=30_000, params=None, seed=0):
        self.N_h, self.N_f = N_h, N_f
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.rng = np.random.default_rng(seed)
        self._init_households()
        self._init_loans()
        self._init_firms()

    # ------------------------------------------------------------------ setup
    def _init_households(self):
        p, rng, N = self.p, self.rng, self.N_h
        self.income0 = rng.lognormal(p["income_mu"], p["income_sigma"], N)   # monthly RM
        self.income0 = np.clip(self.income0, 500, None)
        self.cls = np.where(self.income0 <= DOSM_2022["b40_upper"], 0,
                            np.where(self.income0 < DOSM_2022["t20_lower"], 1, 2))
        self.class_names = np.array(["B40", "M40", "T20"])

        self.lam = np.empty(N)
        self.ess_share = np.empty(N)
        self.fuel_share = np.empty(N)
        buffer = np.empty(N)
        for k, c in enumerate(self.class_names):
            m = self.cls == k
            lo, hi = p["lambda_range"][c]
            self.lam[m] = rng.uniform(lo, hi, m.sum())
            self.ess_share[m] = p["essential_share"][c]
            self.fuel_share[m] = p["fuel_share"][c]
            lo, hi = p["buffer_months"][c]
            buffer[m] = rng.uniform(lo, hi, m.sum())
        self.w = buffer * self.income0                       # liquid wealth, RM
        self.ess0_week = self.ess_share * self.income0 / WEEKS_PER_MONTH
        self.employer = rng.integers(0, self.N_f, N)
        self.defaulted = np.zeros(N, dtype=bool)
        self.default_week = np.full(N, -1)

    def _init_loans(self):
        p, rng, N = self.p, self.rng, self.N_h
        L = len(LOAN_TYPES)
        self.bal = np.zeros((N, L))
        self.n_left = np.zeros((N, L), dtype=int)
        self.fixed_pay = np.zeros((N, L))
        self.floating = np.array([p["loan_floating"][t] for t in LOAN_TYPES])
        self.spread = np.array([p["loan_spread"][t] for t in LOAN_TYPES])
        annual_income = self.income0 * 12
        for j, t in enumerate(LOAN_TYPES):
            prob = np.array([p["loan_prob"][t][c] for c in self.class_names])[self.cls]
            has = rng.random(N) < prob
            lo, hi = p["loan_size"][t]
            self.bal[has, j] = annual_income[has] * rng.uniform(lo, hi, has.sum())
            lo, hi = p["loan_term"][t]
            self.n_left[has, j] = (rng.uniform(lo, hi, has.sum()) * 52).astype(int)
            if not self.floating[j]:   # fixed-rate: payment set at origination (OPR 2.75%)
                rate = self.spread[j]
                self.fixed_pay[has, j] = annuity_payment(self.bal[has, j], rate, self.n_left[has, j])

    def _init_firms(self):
        mu, sd = self.p["firm_capital"]
        self.wf = self.rng.normal(mu, sd, self.N_f)
        self.wf0 = self.wf.copy()
        # firm size = its wage bill; customers' spending reaches firms in proportion to size
        bill = np.bincount(self.employer, weights=self.income0, minlength=self.N_f)
        size = bill + 0.1 * bill.mean()
        self.firm_weight = size / size.sum()

    # --------------------------------------------------------------- dynamics
    def step(self, t, opr, price_level, wage_level, fuel_mult):
        p, rng = self.p, self.rng
        N, Nf = self.N_h, self.N_f

        # 1. Wages: employer firm -> household
        wage = self.income0 * wage_level / WEEKS_PER_MONTH
        self.w += wage
        np.subtract.at(self.wf, self.employer, wage)

        # 2. Open bath: external demand (exports / government) flows into firms
        self.wf += p["external_demand_share"] * wage.sum() * self.firm_weight

        # 3. Essential spending (CPI-indexed, fuel component hit by subsidy removal)
        ess = self.ess0_week * price_level * (1 + self.fuel_share * (fuel_mult - 1))
        self.w -= ess
        self.wf += ess.sum() * self.firm_weight

        # 4. Loan servicing (balance-based; floating loans re-priced at OPR + spread)
        rate = np.where(self.floating, opr + self.spread, self.spread)          # % p.a. per type
        float_pay = annuity_payment(self.bal, rate[None, :], self.n_left)
        pay = np.where(self.floating[None, :], float_pay, self.fixed_pay)
        pay = np.where(self.n_left > 0, np.minimum(pay, self.bal * (1 + rate / 100 / 52)), 0.0)
        interest = self.bal * rate[None, :] / 100 / 52
        self.bal = np.maximum(self.bal + interest - pay, 0.0)
        self.n_left = np.where(self.n_left > 0, self.n_left - 1, 0)
        total_pay = pay.sum(axis=1)
        self.w -= total_pay

        # 5. Unsecured debt (Eq.3): interest on negative liquid wealth at card rate
        neg = self.w < 0
        card_interest = np.zeros(N)
        card_interest[neg] = -self.w[neg] * p["card_rate"] / 100 / 52
        self.w -= card_interest
        bank_inflow = total_pay.sum() + card_interest.sum()
        self.wf += p["bank_recycle_share"] * bank_inflow * self.firm_weight

        # 6. Kinetic exchange (Chakraborti-Chakrabarti, bipartite, 1-to-1 matching)
        #    Only households with positive liquid wealth trade (you cannot spend debt).
        hh = rng.choice(N, Nf, replace=False)
        wh = np.maximum(self.w[hh], 0.0)
        wf = np.maximum(self.wf, 0.0)
        lam_h, lam_f = self.lam[hh], p["lambda_firm"]
        eps = rng.random(Nf)
        pool = (1 - lam_h) * wh + (1 - lam_f) * wf
        new_wh = lam_h * wh + eps * pool
        new_wf = lam_f * wf + (1 - eps) * pool
        self.w[hh] += new_wh - wh
        self.wf += new_wf - wf

        # 6b. Firms pay out aggregate profit (wealth above starting capital) to the bath
        agg_excess = self.wf.sum() - self.wf0.sum()
        if agg_excess > 0:
            excess = np.maximum(self.wf - self.wf0, 0.0)
            self.wf -= p["firm_payout_rate"] * agg_excess * excess / excess.sum()

        # 7. Distress metrics
        monthly_income = self.income0 * wage_level
        monthly_debt_service = (total_pay + card_interest) * WEEKS_PER_MONTH
        dsr = monthly_debt_service / monthly_income
        limit = -p["credit_limit_months"] * monthly_income
        new_def = (~self.defaulted) & (self.w < limit)
        self.defaulted |= new_def
        self.default_week[new_def] = t
        return dsr

    def run(self, scenario="baseline", weeks=260):
        path = scenario_path(scenario, weeks)
        price, wage_level = 1.0, 1.0
        out = {k: np.zeros((weeks, 3)) for k in ("dsr60", "negative", "pd_cum")}
        out["firm_wealth"] = np.zeros(weeks)
        out["firm_distress"] = np.zeros(weeks)
        for t in range(weeks):
            price *= (1 + path["cpi"][t] / 100) ** (1 / 52)
            wage_level *= (1 + path["wage_g"][t] / 100) ** (1 / 52)
            dsr = self.step(t, path["opr"][t], price, wage_level, path["fuel"][t])
            for k in range(3):
                m = self.cls == k
                out["dsr60"][t, k] = np.mean(dsr[m] > self.p["dsr_threshold"])
                out["negative"][t, k] = np.mean(self.w[m] < 0)
                out["pd_cum"][t, k] = np.mean(self.defaulted[m])
            out["firm_wealth"][t] = self.wf.mean()
            out["firm_distress"][t] = np.mean(self.wf < 0)
        out["income0"] = self.income0
        out["cls"] = self.cls
        out["defaulted"] = self.defaulted.copy()
        out["path"] = path
        return out


def monte_carlo(scenarios=SCENARIOS, n_seeds=10, weeks=260, N_h=30_000, N_f=7_500,
                params=None, verbose=True):
    """Run every scenario over the same seeds (common random numbers)."""
    results = {s: [] for s in scenarios}
    for seed in range(n_seeds):
        for s in scenarios:
            m = KineticDebtModel(N_h, N_f, params, seed=seed)
            results[s].append(m.run(s, weeks))
        if verbose:
            print(f"seed {seed + 1}/{n_seeds} done")
    return results
