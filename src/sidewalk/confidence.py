"""Confidence, uncertainty and artefact risk.

Nothing here is a neural-network probability. Every term is something we actually
measured about the cross-section, so a low confidence can always be explained in
one sentence to whoever has to act on the number.

Confidence  -- weighted geometric mean of evidence terms in [0, 1].
Uncertainty -- an error budget in metres, combined in quadrature, turned into a
               95% interval and then used to flag band classifications that are
               too close to call.
Artefact risk -- what to send a human to look at, ranked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CLOSURE_SCORE = {"GOOD": 1.00, "INVESTIGATE": 0.70, "AMBIGUOUS": 0.35, "N/A": 0.60}
BACK_EDGE_SCORE = {"building": 1.00, "general": 0.75, "structure": 0.65,
                   "rail": 0.65, "water": 0.5, "natural": 0.6, "verge": 0.7,
                   "steps": 0.6, "other": 0.5, "void": 0.30}
METHOD_SCORE = {"kerb_normal": 1.00, "fan_fallback": 0.80, "centreline_normal": 0.55}


# --------------------------------------------------------------------------
def sample_uncertainty(df: pd.DataFrame, sigma_survey: float = C.SIGMA_SURVEY) -> pd.DataFrame:
    """Per-sample 1-sigma error budget, in metres.

    Three independent contributions:
      survey   -- relative accuracy of the OS capture itself
      ensemble -- disagreement between neighbouring transects (MAD -> sigma)
      method   -- disagreement between the kerb-normal and fan estimators

    The naive centreline estimator is deliberately excluded here: it is the
    baseline we are trying to beat, not a member of the ensemble.
    """
    d = df.copy()
    sig_ens = 1.4826 * d.ens_mad.fillna(0.0) / np.sqrt(d.ens_n.clip(lower=1))
    sig_meth = (d.w_kerb - d.w_fan).abs() / 2.0
    sig_meth = sig_meth.fillna(0.0).clip(upper=1.5)
    d["sigma_m"] = np.sqrt(sigma_survey ** 2 + sig_ens ** 2 + sig_meth ** 2)
    d["sigma_ensemble"] = sig_ens
    d["sigma_method"] = sig_meth
    return d


def sample_confidence(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Weighted geometric mean of the evidence terms."""
    w = weights or C.CONFIDENCE_WEIGHTS
    d = df.copy()
    # Estimator spread: prefer kerb-normal vs fan. Where the kerb fit was
    # rejected there is no kerb-normal estimate, so fall back to comparing the
    # centreline-normal against the fan rather than penalising the sample twice.
    spread = (d.w_kerb - d.w_fan).abs()
    spread = spread.fillna((d.w_perp - d.w_fan).abs()).fillna(0.30)

    terms = {
        "method": d.method.map(METHOD_SCORE).fillna(0.5),
        "estimator_agreement": np.exp(-spread / 0.50),
        "ensemble_stability": np.exp(-(1.4826 * d.ens_mad.fillna(0.3)) / 0.25),
        "closure": d.get("closure_class", pd.Series("N/A", index=d.index)).map(CLOSURE_SCORE).fillna(0.6),
        "back_edge": d.back_class.map(BACK_EDGE_SCORE).fillna(0.5),
        # Linear ramp: full marks once the cross-section is clear of the nearest
        # other street by JUNCTION_INFLUENCE metres, zero inside the junction.
        "topology": np.clip(d.dist_other_road / C.JUNCTION_INFLUENCE, 0.05, 1.0),
        "source": d.get("osm_score", pd.Series(0.80, index=d.index)).fillna(0.80),
    }
    num = np.zeros(len(d))
    den = 0.0
    for k, v in terms.items():
        wk = w.get(k, 0.0)
        if wk <= 0:
            continue
        num = num + wk * np.log(np.clip(np.asarray(v, dtype=float), 1e-3, 1.0))
        den += wk
        d[f"c_{k}"] = np.asarray(v, dtype=float)
    d["confidence"] = np.exp(num / den) if den else np.nan
    d.loc[~d.valid, "confidence"] = np.nan
    return d


# --------------------------------------------------------------------------
def band_with_uncertainty(width: float, sigma: float, z: float = C.CI_Z):
    """Band classification plus an honest statement about how close it is to call."""
    lo, hi = width - z * sigma, width + z * sigma
    b, b_lo, b_hi = C.band_of(width), C.band_of(max(lo, 0.0)), C.band_of(hi)
    return dict(band=b, ci_low=max(lo, 0.0), ci_high=hi,
                band_borderline=(b_lo != b) or (b_hi != b),
                band_range=f"{b_lo} - {b_hi}" if b_lo != b_hi else b)


def segment_uncertainty(sub: pd.DataFrame, decorrelation_m: float = 5.0,
                        spacing: float = C.SAMPLE_SPACING) -> tuple[float, float]:
    """Segment-level 1-sigma.

    Consecutive 1 m transects are not independent -- they see the same kerb and
    the same building line. We therefore deflate the effective sample size to one
    independent observation per `decorrelation_m` metres rather than pretending
    every metre is a fresh measurement.
    """
    w = sub.width_m.to_numpy(dtype=float)
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return np.nan, np.nan
    span = sub.s.max() - sub.s.min() + spacing
    n_eff = max(span / decorrelation_m, 1.0)
    spread = 1.4826 * np.median(np.abs(w - np.median(w)))
    sig_within = float(np.nanmedian(sub.sigma_m)) if "sigma_m" in sub else C.SIGMA_SURVEY
    sigma = float(np.sqrt(sig_within ** 2 + (spread ** 2) / n_eff))
    return sigma, float(n_eff)


# --------------------------------------------------------------------------
def artefact_risk(segments: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Rank segments by how likely the number is an artefact rather than a pavement.

    The brief asks for the five widest to be checked by eye. Width alone is a poor
    trigger: a genuinely generous pavement scores the same as a ray that escaped
    into a forecourt. This ranks by *evidence of trouble*, so the manual review
    goes where it is actually needed -- and we still report the five widest,
    because that is what was asked for.
    """
    w = weights or C.ARTEFACT_WEIGHTS
    s = segments.copy()
    terms = {
        # risk rises past the 'high capacity' boundary, where footways get rare
        "width_extremity": 1 / (1 + np.exp(-(s.width_median - 3.3) / 0.7)),
        "low_confidence": 1.0 - s.confidence.fillna(0.5),
        "instability": np.clip(s.width_iqr.fillna(0) / 1.0, 0, 1),
        "open_back_edge": (s.back_edge_mode != "building").astype(float),
        "closure_ambiguous": s.get("pct_closure_ambiguous", pd.Series(0.0, index=s.index)).fillna(0) / 100.0,
        "junction_proximity": np.exp(-s.get("mean_dist_other_road", pd.Series(50.0, index=s.index)).fillna(50) / C.JUNCTION_INFLUENCE),
        # Does the strip of ground we claim is pavement actually sit on surveyed
        # footway polygons? Computed by polygon intersection, independently of
        # the ray casting that produced the width.
        "ribbon_mismatch": 1.0 - s.get("ribbon_coverage", pd.Series(1.0, index=s.index)).fillna(0.9),
    }
    total = np.zeros(len(s))
    for k, v in terms.items():
        wk = w.get(k, 0.0)
        total = total + wk * np.clip(np.asarray(v, dtype=float), 0, 1)
        s[f"risk_{k}"] = np.clip(np.asarray(v, dtype=float), 0, 1)
    s["artefact_risk"] = total / sum(w.values())

    def why(row):
        bits = sorted(((w[k] * row[f"risk_{k}"], k) for k in w), reverse=True)
        return ", ".join(k.replace("_", " ") for v, k in bits[:2] if v > 0.05) or "none"

    s["artefact_reason"] = s.apply(why, axis=1)
    return s
