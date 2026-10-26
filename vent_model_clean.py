import math
from typing import NamedTuple

from sympy import exp, nsolve, symbols

# Default parameters to start with
adic = {
    'A': 17.8, 'S0': 0, 'P0': 30, 'Dw': 0, 'K': 863, 'Tc': 31.8, 'Tp': 34.6, 'IE': 0.5,
    't1': 6.28, 't2': 25.5, 't3': 273, 't4': 1.79, 't5': 1.75, 't6': 1.11, 't7': 0.191,
    't8': 42.3, 't9': 9.06, 't10': -0.0156, 't11': -0.00286, 't12': 33.6, 't13': -0.0138,
    't14': 68.3, 't15': 2.10, 't16': 1.30, 't17': 1.42, 't18': 0.68,
    'n1': 0, 'n2': 0, 'n3': 0, 'n4': 0, 'n5': 0, 'n6': 0, 'n7': 0,
    'ageref': 35, 'pmaref': 1860, 'wgtref': 70.0, 'heightref': 1.7, 'sexref': 1,
}

# Run / dosing configuration. Dose 1's rate is supplied per call (doserate_test).
c = {
    'PSV': 7, 'PAVfactor': 0.5, 'xth': 0.25,
    'time': 60,
    'doserate1': 0, 'dosetime1': 0, 'tinf1': 60,
    'doserate2': 0, 'dosetime2': 0, 'tinf2': 0,
    'doserate3': 0, 'dosetime3': 0, 'tinf3': 0,
    'doserate4': 0, 'dosetime4': 0, 'tinf4': 0,
    'doserate5': 0, 'dosetime5': 0, 'tinf5': 0,
}

# Acid-base equilibrium constants
KW = 2.39e-14
KC = 2.45e-11
K3 = 1.16e-10
K2 = 2.19e-7
KH = 1.77e-7

# Handy option set for the "more robust" solver behaviour
MAXSTEPS_KWARGS = {'maxsteps': 50, 'solver': 'mnewton'}


# ---------------------------------------------------------------------------
# PK covariate functions
# ---------------------------------------------------------------------------
def fageing(x, age, ageref):
    return math.exp(x * (age - ageref))


def fsigmoid(x, e50, lam):
    return x ** lam / (x ** lam + e50 ** lam)


def fcentral(weight):
    return fsigmoid(weight, adic['t12'], 1)


def fclmaturation(pma):
    return fsigmoid(pma, adic['t8'], adic['t9'])


def fq3maturation(age):
    """Q3 maturation; `age` is in years and is converted to PMA in weeks."""
    pma_weeks = age * 52 + 40
    return fsigmoid(pma_weeks, adic['t14'], 1)


def fopiates(x, opiates, age):
    return math.exp(x * age) if opiates == 1 else 1


def bmi(weight, height):
    return weight / height ** 2


def falsallami(sex, age, height, weight):
    """Fat-free mass (Al-Sallami). sex: 1 = male, 0 = female."""
    if sex == 1:
        floor, slope, den_base, den_bmi = 0.88, -12.7, 6680, 216
    elif sex == 0:
        floor, slope, den_base, den_bmi = 1.11, -1.1, 8780, 244
    else:
        raise ValueError(f"sex must be 0 or 1, got {sex!r}")

    age_factor = floor + (1 - floor) / (1 + (age / 13.4) ** slope)
    return age_factor * (9270 * weight / (den_base + den_bmi * bmi(weight, height)))


def pharmamodel(sex, age, pma, weight, height, opiates):
    """Return (v1, v2, v3, cl, q2, q3) for the patient."""
    t = adic

    v1_arterial = t['t1'] * (fcentral(weight) / fcentral(t['wgtref'])) * math.exp(t['n1'])
    v1 = v1_arterial * (1 + t['t17'] * (1 - fcentral(weight)))

    v2 = (t['t2'] * (weight / t['wgtref'])
          * fageing(t['t10'], age, t['ageref']) * math.exp(t['n2']))

    ffm_ratio = (falsallami(sex, age, height, weight)
                 / falsallami(t['sexref'], t['ageref'], t['heightref'], t['wgtref']))
    v3 = t['t3'] * ffm_ratio * fopiates(t['t13'], opiates, age) * math.exp(t['n3'])

    cl_typical = t['t4'] if sex == 1 else t['t15']
    cl = (cl_typical * (weight / t['wgtref']) ** 0.75
          * (fclmaturation(pma) / fclmaturation(t['pmaref']))
          * fopiates(t['t11'], opiates, age) * math.exp(t['n4']))

    q2_arterial = (t['t5'] * (v2 / t['t2']) ** 0.75
                   * (1 + t['t16'] * (1 - fq3maturation(age))) * math.exp(t['n5']))
    q2 = q2_arterial * t['t18']

    q3 = (t['t6'] * (v3 / t['t3']) ** 0.75
          * (fq3maturation(age) / fq3maturation(t['ageref'])) * math.exp(t['n6']))

    return v1, v2, v3, cl, q2, q3


# ---------------------------------------------------------------------------
# 3-compartment kinetics
# ---------------------------------------------------------------------------
def threecompvar(v1, v2, v3, cl, q2, q3):
    """Return the three disposition rate constants (alpha, beta, gamma).

    Solves the characteristic cubic with the trigonometric method.
    """
    a0 = (cl * q2 * q3) / (v1 * v2 * v3)
    a1 = ((cl * q3) / (v1 * v3) + (q2 * q3) / (v2 * v3) + (q2 * q3) / (v2 * v1)
          + (cl * q2) / (v1 * v2) + (q3 * q2) / (v3 * v1))
    a2 = cl / v1 + q2 / v1 + q3 / v1 + q2 / v2 + q3 / v3

    p = a1 - a2 ** 2 / 3
    q = 2 * a2 ** 3 / 27 - a1 * a2 / 3 + a0
    r1 = (-(p ** 3) / 27) ** 0.5
    r2 = 2 * r1 ** (1 / 3)
    psi = math.acos(-q / (2 * r1)) / 3

    alpha = -(math.cos(psi) * r2 - a2 / 3)
    beta = -(math.cos(psi + 2 * math.pi / 3) * r2 - a2 / 3)
    gamma = -(math.cos(psi + 4 * math.pi / 3) * r2 - a2 / 3)
    return alpha, beta, gamma


def _infusion_term(rate, t_inf, t_start, t, wgt, coeffs, lambdas):
    """Concentration contribution of one zero-order infusion at time t."""
    elapsed = t - t_start
    if rate == 0 or elapsed <= 0:
        return 0.0

    infusing = min(elapsed, t_inf)       # time spent under infusion so far
    decaying = max(elapsed - t_inf, 0)   # time since the infusion stopped
    total = sum(k * (1 - math.exp(-lam * infusing)) * math.exp(-lam * decaying) / lam
                for k, lam in zip(coeffs, lambdas))
    return rate * wgt * total / 1000


def infusionmulti(alpha, beta, gamma, wgt, v1, v2, v3, q2, q3, t, doses):
    k21 = q2 / v2
    k31 = q3 / v3

    A = (1 / v1) * (k21 - alpha) * (k31 - alpha) / ((alpha - beta) * (beta - gamma))
    B = (1 / v1) * (k21 - beta) * (k31 - beta) / ((beta - alpha) * (beta - gamma))
    C = (1 / v1) * (k21 - gamma) * (k31 - gamma) / ((gamma - beta) * (gamma - alpha))

    coeffs, lambdas = (A, B, C), (alpha, beta, gamma)
    return sum(_infusion_term(rate, t_inf, t_start, t, wgt, coeffs, lambdas)
               for rate, t_inf, t_start in doses)


def _dose_schedule(first_rate, cfg=c):
    """Build [(rate, t_inf, t_start), ...] for doses 1-5 from the config dict."""
    rates = [first_rate] + [cfg[f'doserate{i}'] for i in range(2, 6)]
    return [(rate, cfg[f'tinf{i}'], cfg[f'dosetime{i}'])
            for i, rate in enumerate(rates, start=1)]


# ---------------------------------------------------------------------------
# Respiratory / acid-base model
# ---------------------------------------------------------------------------
class Params(NamedTuple):

    periph_gain: float   # a[0]
    periph_dw: float     # a[1]
    periph_s0: float     # a[2]
    tc: float            # a[3]  central threshold (H+, nmol/L)
    tp: float            # a[4]  peripheral threshold (H+, nmol/L)
    pao2: float          # a[5]
    fr: float            # a[6]  respiratory rate
    vco2: float          # a[7]
    vd: float            # a[8]  dead space
    shunt: float         # a[9]
    resistance: float    # a[10]
    peep: float          # a[11]
    c_lung: float        # a[12]
    c_cw: float          # a[13]
    ie: float            # a[16] I:E ratio
    phosphate: float     # a[21]
    hemoglobin: float    # a[22]
    age: float           # a[25]
    weight: float        # a[26]
    height: float        # a[27]
    sex: int             # a[28]
    opiates: int         # a[29]
    sid_base: float      # a[30]
    vt_ref_weight: float  # a[31] weight used for tidal volume per kg


_PARAM_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  16, 21, 22, 25, 26, 27, 28, 29, 30, 31)


def _unpack_params(a):
    return Params(*(a[i] for i in _PARAM_INDICES))


# Unknowns of the nonlinear system
(sc, sp, dc, dp, vept, vemin, pco2, vetot, vtot, vtmin, vth,
 pmus, tmax, e, ti, vtpt, sid, hplus) = symbols(
    'sc sp dc dp vept vemin pco2 vetot vtot vtmin vth pmus tmax e ti vtpt sid hplus')

SYMS_THRESHOLD = [sc, sp, dc, dp, vept, vemin, pco2, vetot, vtot, vtmin, vth,
                  pmus, tmax, e, ti, vtpt, sid, hplus]                 # 18 unknowns
SYMS_NO_THRESHOLD = [s for s in SYMS_THRESHOLD if s not in (vth, tmax)]  # 16 unknowns

INITIAL_GUESS = {sc: 1, sp: 1, dc: 2, dp: 2, vept: 3, vemin: 3, pco2: 30, vetot: 6,
                 vtot: 5, vtmin: 1, vth: 0.2, pmus: 3, tmax: 0.3, e: 20, ti: 1,
                 vtpt: 1, sid: 0, hplus: 0.00000004}

#   1: h+ > tc and h+ > tp   2: h+ < tc, h+ > tp   3: h+ > tc, h+ < tp   4: neither
CASES = [(True, True), (False, True), (True, False), (False, False)]


def _build_system(p, psv_test, vco2, cprop, xth, c_active, p_active, with_threshold):

    R = p.resistance
    hb = p.hemoglobin
    vth_term = vth if with_threshold else xth

    # chemoreflex sensitivities and drives
    if c_active or p_active:
        eq_sc = 1.0589 * (-0.4692 * cprop + 1.4919) - sc
        eq_sp = p.periph_s0 + p.periph_gain / (p.pao2 - p.periph_dw) - sp
    else:
        eq_sc = sc
        eq_sp = sp
    eq_dc = sc * (1000000000 * hplus - p.tc) - dc if c_active else dc
    eq_dp = sp * (1000000000 * hplus - p.tp) - dp if p_active else dp

    # ventilation
    eq_vept = dc + dp - vept
    eq_vemin = vtmin * p.fr - vemin
    eq_vtmin = (psv_test - vth_term * R) / e - vtmin
    eq_vetot = vemin + vept - vetot
    eq_vtot = vetot / p.fr - vtot
    eq_pco2 = (0.863 * vco2) / (pco2 * (1 - p.vd / vtot - p.shunt)) - vetot
    eq_elast = 1 / p.c_cw + 1 / p.c_lung - e

    eqs = [eq_sc, eq_sp, eq_dc, eq_dp, eq_vept, eq_vemin, eq_vtmin, eq_vetot,
           eq_vtot, eq_pco2, eq_elast]

    # flow-threshold equations (only when vth/tmax are unknowns)
    if with_threshold:
        eq_tmax = (psv_test * (-e * exp(-e * tmax / R) / R + exp(-tmax / 0.15) / 0.15)
                   / (R * (-e * R / 0.15 + 1))
                   + 2 * pmus * ti * (e * (1 + R * ti / e) * exp(-e * tmax / R) / R - 1 / ti) / e)
        eq_vth = (xth * ((psv_test / R) * (exp(-tmax * e / R) - exp(-tmax / 0.15))
                         / (1 - R / 0.15 * e)
                         + (2 * pmus / e * ti)
                         * ((1 + R / e * ti) * (1 - exp(-tmax * e / R) - tmax / ti)))
                  - vth)
        eqs += [eq_tmax, eq_vth]

    # respiratory mechanics
    eq_vtpt = vept / p.fr - vtpt
    eq_ti = p.ie / (1 + p.ie) * (60 / p.fr) - ti
    eq_pmus = vtpt / ti * R + vtpt * e + p.peep - pmus

    # acid-base
    eq_sid = (10000 * p.sid_base + 28 * hb + 18000 * p.phosphate) / 10000 - sid
    eq_h = (sid + hplus - KC * pco2 / hplus
            - ((21 * hb * 10 / 66500) - hplus * (16 * hb * 10 / 66500) / (KH + hplus))
            - 2 * K3 * KC * pco2 / hplus ** 2
            - KW / hplus
            - p.phosphate * (2 - hplus / (K2 + hplus)))

    eqs += [eq_vtpt, eq_ti, eq_pmus, eq_sid, eq_h]
    return eqs


def _solve_all_cases(p, psv_test, vco2, cprop, xth, with_threshold, nsolve_kwargs):
    """Solve the four cases; each solution is a {name: value} dict (zeros on failure)."""
    syms = SYMS_THRESHOLD if with_threshold else SYMS_NO_THRESHOLD
    guess = [INITIAL_GUESS[s] for s in syms]
    names = [s.name for s in syms]

    solutions = []
    for c_active, p_active in CASES:
        eqs = _build_system(p, psv_test, vco2, cprop, xth, c_active, p_active, with_threshold)
        try:
            values = list(nsolve(eqs, syms, guess, **nsolve_kwargs))
        except Exception:
            values = [0] * len(syms)
        solutions.append(dict(zip(names, values)))
    return solutions


def _select_solution(s1, s2, s3, s4):
    if s1['dc'] >= 0 and s1['dp'] >= 0:
        return s1
    if s1['dc'] < 0 and s1['dp'] > 0 and s2['dp'] > 0:
        return s2
    if s1['dc'] > 0 and s1['dp'] < 0 and s3['dc'] > 0:
        return s3
    return s4


def modelevaluationfcn(a, psv_test, doserate_test, xth, ecco, fr_adjust, nsolve_kwargs=None):
    """Evaluate the model for one pressure-support / dose-rate setting.

    nsolve_kwargs: extra keyword args for the sympy nsolve calls of the main
        systems. None -> default solver; MAXSTEPS_KWARGS -> maxsteps=50, 'mnewton'.

    Returns [plstat, pldyn, pes, ph, sid, cprop, vtot, pco2, vtmin, elastance, vt_per_kg].
    `a` is not modified.
    """
    nsolve_kwargs = nsolve_kwargs or {}

    p = _unpack_params(a)
    if fr_adjust != 0:
        p = p._replace(fr=fr_adjust)
    if psv_test <= xth * p.resistance:
        xth = 0
    vco2 = p.vco2 - ecco

    # ---- propofol concentration --------------------------------------------
    pma = p.age * 52 + 39
    v1, v2, v3, cl, q2, q3 = pharmamodel(p.sex, p.age, pma, p.weight, p.height, p.opiates)
    alpha, beta, gamma = threecompvar(v1, v2, v3, cl, q2, q3)
    cprop = infusionmulti(alpha, beta, gamma, p.weight, v1, v2, v3, q2, q3,
                          c['time'], _dose_schedule(doserate_test))

    # ---- main solve (with flow threshold) -----------------------------------
    solution = _select_solution(
        *_solve_all_cases(p, psv_test, vco2, cprop, xth, True, nsolve_kwargs))

    # ---- fallback if the flow-threshold solve failed --------------------------
    if all(v == 0 for v in solution.values()):
        vth_norm = xth
        solution = _select_solution(
            *_solve_all_cases(p, psv_test, vco2, cprop, vth_norm, False, nsolve_kwargs))
        solution = {**solution, 'vth': vth_norm, 'tmax': 0.3}

    # ---- static / dynamic compliance pressures (closed-form) ------------------
    plstat = solution['vtot'] / p.c_lung
    pes = solution['pmus'] - solution['vtot'] / p.c_cw
    pes_act = 0 if pes < 0 else -pes
    pldyn = psv_test - pes_act

    # ---- pH ---------------------------------------------------------------------
    ph = -math.log10(solution['hplus'])

    return [plstat, pldyn, pes, ph, solution['sid'], cprop,
            solution['vtot'], solution['pco2'], solution['vtmin'], solution['e'],
            solution['vtot'] * 1000 / p.vt_ref_weight]