import copy
import numpy as np

# =============================================================================
# FYP: Modelling Debt Risk on Different Income Levels Using Econophysics
# Open-Bath Bipartite Kinetic Exchange Model -- v3
#
# v2 (fixes to the original notebook):
#   wages paid weekly; loans carry balances re-priced at OPR + spread; loans
#   depend on income class; DOSM 2022 income calibration; multi-seed runs.
# v3 (this version):
#   1. Income shocks: weekly job-loss hazard by class; workers at insolvent
#      firms face a higher hazard (firm failure -> layoffs -> household stress).
#   2. New borrowing: overdrafts are refinanced into personal loans, and loans
#      are renewed when they mature, both capped by a bank DSR limit.
#   3. Default = 90 days past due (13 weeks with the credit line exhausted),
#      as banks define impaired loans. Defaulted debt is written off.
#   4. Warm-up (burn-in) period before the scenario clock starts.
#   5. A money ledger (the "bath") so conservation can be checked every step.
#   6. Starting loans respect the same bank DSR cap as new loans.
# =============================================================================

WEEKS_PER_MONTH = 52 / 12

# DOSM HIES 2022 (gross monthly household income, RM)
DOSM_2022 = dict(
    median=6338, mean=8479,
    b40_upper=5249, t20_lower=11820,
    class_mean={"B40": 3401, "M40": 7971, "T20": 19752},
    income_share={"B40": 0.161, "M40": 0.376, "T20": 0.463},
    gini=0.404,
)

LOAN_TYPES = ["mortgage", "hire_purchase", "personal"]
CLASSES = ["B40", "M40", "T20"]

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

    # --- Existing loans at t=0: take-up probability by class ---
    loan_prob={
        "mortgage":      {"B40": 0.20, "M40": 0.40, "T20": 0.55},
        "hire_purchase": {"B40": 0.35, "M40": 0.45, "T20": 0.45},
        "personal":      {"B40": 0.30, "M40": 0.20, "T20": 0.10},
    },
    loan_size={"mortgage": (2.5, 4.5), "hire_purchase": (0.4, 1.0), "personal": (0.2, 0.6)},  # x annual income
    loan_term={"mortgage": (5, 30), "hire_purchase": (2, 9), "personal": (2, 10)},           # years remaining
    new_loan_term={"mortgage": 30, "hire_purchase": 7, "personal": 5},                         # years, new loans
    loan_floating={"mortgage": True, "hire_purchase": False, "personal": True},
    loan_spread={"mortgage": 1.50, "hire_purchase": 3.00, "personal": 5.00},   # % p.a. over OPR (HP: fixed rate)

    # --- New borrowing ---
    bank_dsr_cap=0.60,            # banks approve new credit only if DSR stays below this
    initial_dsr_cap=0.60,         # cap on starting loans; raise it if data show legacy borrowers above 60%
    renew_prob={"mortgage": 0.50, "hire_purchase": 0.80, "personal": 0.50},  # take a new loan when one matures
    new_loan_annual={             # annual chance a household WITHOUT this loan takes one out
        "mortgage":      {"B40": 0.035, "M40": 0.08, "T20": 0.13},
        "hire_purchase": {"B40": 0.10, "M40": 0.12, "T20": 0.12},
        "personal":      {"B40": 0.08, "M40": 0.05, "T20": 0.03},
    },
    refinance_topup_months=0.5,   # extra cash on top of clearing the overdraft
    min_new_loan=1_000.0,         # RM; smaller approvals are not worth booking

    # --- Unsecured overdraft / card debt and default ---
    card_rate=15.0,               # % p.a., not OPR-linked
    credit_limit_months=2.0,      # credit line = this many months of income
    default_weeks=13,             # weeks with the line exhausted before default (~90 days past due)

    # --- Income shocks ---
    job_loss_annual={"B40": 0.06, "M40": 0.04, "T20": 0.025},   # annual probability of an income-loss spell
    unemployment_weeks={"B40": 20, "M40": 16, "T20": 12},        # mean spell length
    income_cut=0.50,              # share of household income lost during a spell (one of ~1.8 earners)
    layoff_multiplier=3.0,        # hazard multiplier for workers at firms with negative wealth

    # --- Firms and the open bath ---
    firm_capital=(50_000, 10_000),
    external_demand_share=0.15,   # exports/government demand, as a share of the wage bill
    bank_recycle_share=1.0,       # share of loan repayments re-lent to firms
    firm_payout_rate=0.05,        # weekly share of aggregate firm profit paid out to the bath

    dsr_threshold=0.60,           # reporting threshold for "high DSR"
    burn_in_weeks=26,             # warm-up under baseline conditions before measuring
)


def scenario_path(name, weeks):
    """Weekly paths: OPR (% p.a.), CPI (annual %), wage growth (annual %), fuel multiplier, job-loss multiplier."""
    t = np.arange(weeks)
    opr = np.full(weeks, 2.75)
    cpi = np.full(weeks, 2.0)
    wage_g = np.full(weeks, 3.0)
    fuel = np.ones(weeks)
    jobloss = np.ones(weeks)

    if name in ("opr_hike", "combined"):
        for wk in (13, 26, 39, 52):          # four 25bp hikes -> 3.75%
            opr[t >= wk] += 0.25
    if name in ("subsidy_removal", "combined"):
        fuel[t >= 26] = 2.60 / 1.99           # unsubsidised / BUDI95 RON95 price
        cpi[(t >= 26) & (t < 78)] += 1.0      # one-year pass-through to CPI
    if name in ("inflation_spike", "combined"):
        cpi[(t >= 26) & (t < 130)] = np.maximum(cpi[(t >= 26) & (t < 130)], 6.0)
    if name == "recession":                    # optional extra scenario
        jobloss[(t >= 26) & (t < 104)] = 2.0
        wage_g[(t >= 26) & (t < 104)] = 0.0
    return dict(opr=opr, cpi=cpi, wage_g=wage_g, fuel=fuel, jobloss=jobloss)


SCENARIOS = ["baseline", "opr_hike", "subsidy_removal", "inflation_spike", "combined"]


def annuity_payment(balance, annual_rate_pct, n_weeks):
    r = np.asarray(annual_rate_pct, float) / 100 / 52
    n = np.maximum(n_weeks, 1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pay = np.where(r > 0, balance * r / (1 - (1 + r) ** (-n)), balance / n)
    return np.where(n_weeks > 0, pay, 0.0)


def annuity_factor(annual_rate_pct, n_weeks):
    """Payment per RM1 of principal."""
    return annuity_payment(np.ones_like(np.asarray(n_weeks, float)), annual_rate_pct, n_weeks)


class KineticDebtModel:
    def __init__(self, N_h=120_000, N_f=30_000, params=None, seed=0):
        self.N_h, self.N_f = N_h, N_f
        self.p = copy.deepcopy(DEFAULT_PARAMS)
        if params:
            self.p.update(copy.deepcopy(params))
        self.rng = np.random.default_rng(seed)
        self.bath = 0.0          # running ledger of money that left (+) / entered (-) the system
        self._init_households()
        self._init_loans()
        self._init_firms()
        self.initial_money = self.total_money()

    # ------------------------------------------------------------------ setup
    def _by_class(self, table, draw_range=False):
        out = np.empty(self.N_h)
        for k, c in enumerate(CLASSES):
            m = self.cls == k
            v = table[c]
            out[m] = self.rng.uniform(*v, m.sum()) if draw_range else v
        return out

    def _init_households(self):
        p, rng, N = self.p, self.rng, self.N_h
        self.income0 = np.clip(rng.lognormal(p["income_mu"], p["income_sigma"], N), 500, None)
        self.cls = np.where(self.income0 <= DOSM_2022["b40_upper"], 0,
                            np.where(self.income0 < DOSM_2022["t20_lower"], 1, 2))
        self.lam = self._by_class(p["lambda_range"], draw_range=True)
        self.ess_share = self._by_class(p["essential_share"])
        self.fuel_share = self._by_class(p["fuel_share"])
        self.w = self._by_class(p["buffer_months"], draw_range=True) * self.income0
        self.ess0_week = self.ess_share * self.income0 / WEEKS_PER_MONTH

        self.job_hazard = 1 - (1 - self._by_class(p["job_loss_annual"])) ** (1 / 52)
        self.exit_prob = 1 / self._by_class(p["unemployment_weeks"])
        # start at the steady-state share in an income-loss spell: h / (h + exit rate)
        self.unemployed = rng.random(N) < self.job_hazard / (self.job_hazard + self.exit_prob)

        self.employer = rng.integers(0, self.N_f, N)
        self.arrears = np.zeros(N, dtype=int)
        self.defaulted = np.zeros(N, dtype=bool)
        self.default_week = np.full(N, -1)

    def _init_loans(self):
        p, rng, N = self.p, self.rng, self.N_h
        L = len(LOAN_TYPES)
        self.bal = np.zeros((N, L))
        self.n_left = np.zeros((N, L), dtype=int)
        self.fixed_pay = np.zeros((N, L))
        self.matured_pending = np.zeros((N, L), dtype=bool)   # loans matured since last borrowing check
        self.floating = np.array([p["loan_floating"][t] for t in LOAN_TYPES])
        self.spread = np.array([p["loan_spread"][t] for t in LOAN_TYPES], float)
        self.renew_prob = np.array([p["renew_prob"][t] for t in LOAN_TYPES])
        annual_income = self.income0 * 12
        for j, t in enumerate(LOAN_TYPES):
            prob = np.array([p["loan_prob"][t][c] for c in CLASSES])[self.cls]
            has = rng.random(N) < prob
            self.bal[has, j] = annual_income[has] * rng.uniform(*p["loan_size"][t], has.sum())
            self.n_left[has, j] = (rng.uniform(*p["loan_term"][t], has.sum()) * 52).astype(int)
            if not self.floating[j]:
                self.fixed_pay[has, j] = annuity_payment(self.bal[has, j], self.spread[j], self.n_left[has, j])

        # Existing loans were approved under a bank DSR cap (by default the same one as new
        # loans): scale down any household whose starting instalments exceed it.
        pay0 = self._scheduled_payments(self._rates(2.75)).sum(axis=1)
        dsr0 = pay0 * WEEKS_PER_MONTH / self.income0
        scale = np.minimum(1.0, p["initial_dsr_cap"] / np.maximum(dsr0, 1e-12))
        self.bal *= scale[:, None]
        self.fixed_pay *= scale[:, None]

    def _init_firms(self):
        mu, sd = self.p["firm_capital"]
        self.wf = self.rng.normal(mu, sd, self.N_f)
        self.wf0 = self.wf.copy()
        bill = np.bincount(self.employer, weights=self.income0, minlength=self.N_f)
        size = bill + 0.1 * bill.mean()
        self.firm_weight = size / size.sum()

    def total_money(self):
        return self.w.sum() + self.wf.sum() + self.bath

    # ---------------------------------------------------------------- helpers
    def _rates(self, opr):
        return np.where(self.floating, opr + self.spread, self.spread)

    def _scheduled_payments(self, rate):
        float_pay = annuity_payment(self.bal, rate[None, :], self.n_left)
        pay = np.where(self.floating[None, :], float_pay, self.fixed_pay)
        return np.where(self.n_left > 0, np.minimum(pay, self.bal * (1 + rate / 100 / 52)), 0.0)

    def _originate(self, idx, j, principal, rate_j, term_weeks, to_household):
        """Book a new loan of type j for households idx (principal already DSR-checked)."""
        if idx.size == 0:
            return
        self.bal[idx, j] += principal
        self.n_left[idx, j] = term_weeks
        if not self.floating[j]:
            self.fixed_pay[idx, j] = annuity_payment(self.bal[idx, j], rate_j, term_weeks)
        if to_household:
            self.w[idx] += principal
        else:                                   # purchase price goes to the selling firms
            self.wf += principal.sum() * self.firm_weight
        self.bath -= principal.sum()            # new bank credit enters the system

    def _dsr_headroom(self, idx, monthly_income, current_pay_week):
        """Extra weekly payment each household could take on under the bank DSR cap."""
        cap_week = self.p["bank_dsr_cap"] * monthly_income[idx] / WEEKS_PER_MONTH
        return np.maximum(cap_week - current_pay_week[idx], 0.0)

    # --------------------------------------------------------------- dynamics
    def step(self, t, opr, price_level, wage_level, fuel_mult, jobloss_mult=1.0):
        p, rng = self.p, self.rng
        N, Nf = self.N_h, self.N_f
        alive = ~self.defaulted

        # 0. Income shocks: job-loss spells start and end
        firm_distress = self.wf[self.employer] < 0
        hazard = self.job_hazard * jobloss_mult * np.where(firm_distress, p["layoff_multiplier"], 1.0)
        start = (~self.unemployed) & (rng.random(N) < hazard)
        end = self.unemployed & (rng.random(N) < self.exit_prob)
        self.unemployed = (self.unemployed | start) & ~end

        # 1. Wages: employer firm -> household (reduced during an income-loss spell)
        full_monthly = self.income0 * wage_level
        monthly_income = full_monthly * np.where(self.unemployed, 1 - p["income_cut"], 1.0)
        wage = monthly_income / WEEKS_PER_MONTH
        self.w += wage
        np.subtract.at(self.wf, self.employer, wage)

        # 2. Open bath: external demand (exports / government) flows into firms
        ext = p["external_demand_share"] * (full_monthly / WEEKS_PER_MONTH).sum()
        self.wf += ext * self.firm_weight
        self.bath -= ext

        # 3. Essential spending (CPI-indexed; fuel component hit by subsidy removal)
        ess = self.ess0_week * price_level * (1 + self.fuel_share * (fuel_mult - 1))
        self.w -= ess
        self.wf += ess.sum() * self.firm_weight

        # 4. Loan servicing (balance-based; floating loans re-priced at OPR + spread)
        rate = self._rates(opr)
        pay = self._scheduled_payments(rate)
        interest = self.bal * rate[None, :] / 100 / 52
        self.bal = np.maximum(self.bal + interest - pay, 0.0)
        self.matured_pending |= (self.n_left == 1)
        self.n_left = np.where(self.n_left > 0, self.n_left - 1, 0)
        self.bal[self.n_left == 0] = 0.0
        total_pay = pay.sum(axis=1)
        self.w -= total_pay

        # 5. Unsecured debt (Eq.3): card interest on negative liquid wealth
        card_interest = np.where(self.w < 0, -self.w * p["card_rate"] / 100 / 52, 0.0)
        self.w -= card_interest
        bank_inflow = total_pay.sum() + card_interest.sum()
        recycled = p["bank_recycle_share"] * bank_inflow
        self.wf += recycled * self.firm_weight
        self.bath += bank_inflow - recycled

        # 6. New borrowing (monthly): renew matured loans, refinance overdrafts
        if t % 4 == 0:
            self._new_borrowing(opr, rate, monthly_income, alive & ~self.unemployed, self.matured_pending)
            self.matured_pending[:] = False

        # 7. Kinetic exchange (Chakraborti-Chakrabarti, bipartite, 1-to-1 matching)
        hh = rng.choice(N, Nf, replace=False)
        wh = np.maximum(self.w[hh], 0.0)
        wf = np.maximum(self.wf, 0.0)
        lam_h, lam_f = self.lam[hh], p["lambda_firm"]
        eps = rng.random(Nf)
        pool = (1 - lam_h) * wh + (1 - lam_f) * wf
        self.w[hh] += lam_h * wh + eps * pool - wh
        self.wf += lam_f * wf + (1 - eps) * pool - wf

        # 8. Firms pay out aggregate profit above starting capital to the bath
        agg_excess = self.wf.sum() - self.wf0.sum()
        if agg_excess > 0:
            excess = np.maximum(self.wf - self.wf0, 0.0)
            out = p["firm_payout_rate"] * agg_excess * excess / excess.sum()
            self.wf -= out
            self.bath += out.sum()

        # 9. Arrears and default (~90 days past due), then write-off
        limit = -p["credit_limit_months"] * monthly_income
        exhausted = alive & (self.w < limit)
        self.arrears = np.where(exhausted, self.arrears + 1, 0)
        new_def = alive & (self.arrears >= p["default_weeks"])
        if new_def.any():
            self.defaulted |= new_def
            self.default_week[new_def] = t
            self.bal[new_def] = 0.0             # bank writes off the loans
            self.n_left[new_def] = 0
            self.fixed_pay[new_def] = 0.0
            self.matured_pending[new_def] = False
            written_off = -np.minimum(self.w[new_def], 0.0)
            self.w[new_def] += written_off      # overdraft written off (AKPK / insolvency)
            self.bath -= written_off.sum()

        # 10. Metrics: DSR = debt payments / current gross income (living costs excluded)
        dsr = (total_pay + card_interest) * WEEKS_PER_MONTH / np.maximum(monthly_income, 1.0)
        return dsr

    def _new_borrowing(self, opr, rate, monthly_income, eligible, matured):
        p, rng = self.p, self.rng
        current_pay = self._scheduled_payments(rate).sum(axis=1)

        # (a) Renew matured loans, and first-time borrowing by households without this loan type
        for j, t in enumerate(LOAN_TYPES):
            annual = np.array([p["new_loan_annual"][t][c] for c in CLASSES])[self.cls]
            new_prob = 1 - (1 - annual) ** (4 / 52)
            u = rng.random(self.N_h)
            renew = matured[:, j] & (u < self.renew_prob[j])
            first = (~matured[:, j]) & (self.bal[:, j] <= 0) & (u < new_prob)
            idx = np.flatnonzero(eligible & (renew | first))
            if idx.size == 0:
                continue
            term = int(p["new_loan_term"][t] * 52)
            desired = monthly_income[idx] * 12 * rng.uniform(*p["loan_size"][t], idx.size)
            max_principal = self._dsr_headroom(idx, monthly_income, current_pay) / annuity_factor(rate[j], np.full(idx.size, term))
            principal = np.minimum(desired, max_principal)
            ok = principal >= p["min_new_loan"]
            self._originate(idx[ok], j, principal[ok], rate[j], term, to_household=(t == "personal"))
            current_pay[idx[ok]] += principal[ok] * annuity_factor(rate[j], np.full(ok.sum(), term))

        # (b) Refinance overdrafts into a personal loan (debt consolidation)
        j = LOAN_TYPES.index("personal")
        idx = np.flatnonzero(eligible & (self.w < 0))
        if idx.size == 0:
            return
        term = int(p["new_loan_term"]["personal"] * 52)
        desired = -self.w[idx] + p["refinance_topup_months"] * monthly_income[idx]
        old_pay = annuity_payment(self.bal[idx, j], rate[j], self.n_left[idx, j])
        headroom = self._dsr_headroom(idx, monthly_income, current_pay) + old_pay   # old personal loan is folded in
        max_total = headroom / annuity_factor(rate[j], np.full(idx.size, term))
        principal = np.minimum(desired, max_total - self.bal[idx, j])
        ok = principal >= p["min_new_loan"]
        self._originate(idx[ok], j, principal[ok], rate[j], term, to_household=True)

    # -------------------------------------------------------------------- run
    def run(self, scenario="baseline", weeks=260, check_conservation=False):
        p = self.p
        burn = p["burn_in_weeks"]
        base = scenario_path("baseline", burn) if burn else None
        path = scenario_path(scenario, weeks)
        price, wage_level = 1.0, 1.0

        out = {k: np.zeros((weeks, 3)) for k in ("dsr60", "negative", "pd_cum", "unemployed", "debt_to_income")}
        out["firm_wealth"] = np.zeros(weeks)
        out["firm_distress"] = np.zeros(weeks)
        out["money_error"] = np.zeros(weeks)

        for t in range(burn):                  # warm-up under baseline conditions
            price *= (1 + base["cpi"][t] / 100) ** (1 / 52)
            wage_level *= (1 + base["wage_g"][t] / 100) ** (1 / 52)
            self.step(t - burn, base["opr"][t], price, wage_level, base["fuel"][t], base["jobloss"][t])
        at_risk = ~self.defaulted               # PD is measured on households not in default at t=0
        class_masks = [(self.cls == k) & at_risk for k in range(3)]

        for t in range(weeks):
            price *= (1 + path["cpi"][t] / 100) ** (1 / 52)
            wage_level *= (1 + path["wage_g"][t] / 100) ** (1 / 52)
            dsr = self.step(t, path["opr"][t], price, wage_level, path["fuel"][t], path["jobloss"][t])
            annual_inc = self.income0 * wage_level * 12
            debt = self.bal.sum(axis=1) + np.maximum(-self.w, 0.0)
            for k, m in enumerate(class_masks):
                out["dsr60"][t, k] = np.mean(dsr[m] > p["dsr_threshold"])
                out["negative"][t, k] = np.mean(self.w[m] < 0)
                out["pd_cum"][t, k] = np.mean(self.defaulted[m])
                out["unemployed"][t, k] = np.mean(self.unemployed[m])
                out["debt_to_income"][t, k] = debt[m].sum() / annual_inc[m].sum()
            out["firm_wealth"][t] = self.wf.mean()
            out["firm_distress"][t] = np.mean(self.wf < 0)
            if check_conservation:
                out["money_error"][t] = abs(self.total_money() - self.initial_money) / self.initial_money

        out["income0"] = self.income0
        out["cls"] = self.cls
        out["at_risk"] = at_risk
        out["defaulted"] = self.defaulted & at_risk
        out["path"] = path
        return out


def monte_carlo(scenarios=SCENARIOS, n_seeds=10, weeks=260, N_h=30_000, N_f=7_500,
                params=None, verbose=True):
    """Run every scenario over the same seeds (common random numbers)."""
    results = {s: [] for s in scenarios}
    for seed in range(n_seeds):
        for s in scenarios:
            results[s].append(KineticDebtModel(N_h, N_f, params, seed=seed).run(s, weeks))
        if verbose:
            print(f"seed {seed + 1}/{n_seeds} done")
    return results


# ============================================================================
# Sensitivity analysis
# ============================================================================
SENSITIVITY_PARAMS = [
    "lambda_range", "essential_share", "buffer_months", "loan_prob", "loan_size",
    "loan_spread", "card_rate", "credit_limit_months", "bank_dsr_cap", "initial_dsr_cap", "renew_prob", "new_loan_annual",
    "job_loss_annual", "unemployment_weeks", "income_cut", "external_demand_share",
]


def _scale(value, f):
    if isinstance(value, dict):
        return {k: _scale(v, f) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_scale(v, f) for v in value)
    if isinstance(value, (bool, np.bool_)):
        return value
    return value * f


def _clip_probabilities(params):
    for key in ("lambda_range", "loan_prob", "renew_prob", "new_loan_annual", "job_loss_annual", "essential_share"):
        params[key] = _clip_dict(params[key], 0.0, 0.99)
    params["income_cut"] = min(params["income_cut"], 0.99)
    return params


def _clip_dict(d, lo, hi):
    if isinstance(d, dict):
        return {k: _clip_dict(v, lo, hi) for k, v in d.items()}
    if isinstance(d, tuple):
        return tuple(min(max(v, lo), hi) for v in d)
    return min(max(d, lo), hi)


def sensitivity(names=SENSITIVITY_PARAMS, factor=0.2, scenario="baseline", n_seeds=4,
                weeks=260, N_h=15_000, N_f=3_750, verbose=True):
    """One-at-a-time sensitivity: scale each parameter by (1 - factor) and (1 + factor).

    Returns {name: (pd_low, pd_high)} as arrays of class PDs (seed-averaged), plus the base PD.
    """
    def mean_pd(params):
        return np.mean([KineticDebtModel(N_h, N_f, params, seed=s).run(scenario, weeks)["pd_cum"][-1]
                        for s in range(n_seeds)], axis=0)

    base_pd = mean_pd(None)
    out = {}
    for name in names:
        lo_params = _clip_probabilities({**copy.deepcopy(DEFAULT_PARAMS), name: _scale(DEFAULT_PARAMS[name], 1 - factor)})
        hi_params = _clip_probabilities({**copy.deepcopy(DEFAULT_PARAMS), name: _scale(DEFAULT_PARAMS[name], 1 + factor)})
        out[name] = (mean_pd(lo_params), mean_pd(hi_params))
        if verbose:
            print(f"{name:24s} low={np.round(out[name][0] * 100, 2)}  high={np.round(out[name][1] * 100, 2)}")
    return base_pd, out


# ============================================================================
# Automated sanity checks
# ============================================================================
def run_checks(N_h=10_000, N_f=2_500, seeds=3, verbose=True):
    """Returns a list of (check name, passed, detail)."""
    results = []

    # 1. A single loan amortises to zero exactly at the end of its term
    bal, n, rate = 100_000.0, 30 * 52, 4.25
    pay = float(annuity_payment(np.array(bal), rate, np.array(n)))
    b = bal
    for _ in range(n):
        b = b * (1 + rate / 100 / 52) - pay
    results.append(("Loan amortises to zero at maturity", abs(b) < 1e-6 * bal, f"residual RM{b:.6f}"))

    # 2. Money is conserved (households + firms + bath ledger constant)
    m = KineticDebtModel(N_h, N_f, seed=0)
    out = m.run("combined", 260, check_conservation=True)
    err = out["money_error"].max()
    results.append(("Money conserved every week", err < 1e-9, f"max relative error {err:.2e}"))

    # 3. No NaN or infinite values
    finite = all(np.isfinite(out[k]).all() for k in ("dsr60", "negative", "pd_cum", "debt_to_income"))
    results.append(("No NaN / infinite values", finite, ""))

    # 4. Income calibration matches DOSM 2022 class means within 5%
    m = KineticDebtModel(60_000, 15_000, seed=0)
    errs = [abs(m.income0[m.cls == k].mean() / DOSM_2022["class_mean"][c] - 1) for k, c in enumerate(CLASSES)]
    results.append(("Class mean incomes within 5% of DOSM 2022", max(errs) < 0.05,
                    ", ".join(f"{c} {e:.1%}" for c, e in zip(CLASSES, errs))))

    # 5. Each shock raises financial stress (time-averaged share of households in overdraft),
    #    and the combined shock raises default risk. Paired seeds.
    runs = {s: [KineticDebtModel(N_h, N_f, seed=seed).run(s, 260) for seed in range(seeds)] for s in SCENARIOS}
    def stress(s):
        return np.mean([o["negative"].mean() for o in runs[s]])
    def overall_pd(s):
        return np.mean([o["defaulted"].sum() / o["at_risk"].sum() for o in runs[s]])
    base = stress("baseline")
    for s in SCENARIOS[1:]:
        v = stress(s)
        results.append((f"Shock raises overdraft stress: {s}", v > base, f"{base:.2%} -> {v:.2%}"))
    b, c = overall_pd("baseline"), overall_pd("combined")
    results.append(("Combined shock raises default risk", c > b, f"{b:.2%} -> {c:.2%}"))

    # 6. Higher OPR -> higher instalments on floating loans
    m = KineticDebtModel(2_000, 500, seed=0)
    p_lo = m._scheduled_payments(m._rates(2.75)).sum()
    p_hi = m._scheduled_payments(m._rates(3.75)).sum()
    results.append(("OPR hike raises floating instalments", p_hi > p_lo, f"+{p_hi / p_lo - 1:.1%}"))

    if verbose:
        for name, ok, detail in results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} checks passed")
    return results
