"""
Stage 3: BASELINE V1, then scheme.

BASELINE V1 = Vegas + expected team volume + expected player opportunity
              + basic player efficiency.

Role and efficiency are modelled separately and multiplied:
    points_hat = opportunities_hat * points_per_opportunity_hat

Scheme enters ONLY the efficiency model, as a single points-per-opportunity
term produced by a multivariable, partially pooled sensitivity estimate.
Independent per-condition deltas are not used anywhere: motion, play action,
RPO, screen and alignment are estimated jointly so a shared play-level effect
cannot be counted more than once.

Validation is walk-forward by season. Sensitivities are fit on training
seasons only.
"""
import numpy as np, polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from features import FEATURES_ROLE, FEATURES_EFF, FEATURES_SCHEME_TEAM, CONDITIONS

CHANNELS = ["qb_pass", "qb_rush", "rb_rush", "rb_rec", "wrte_rec"]
TEST_SEASONS = [2024, 2025]
MIN_HIST = 3


# ------------------------------------------------------------------ utilities
def prep(X, medians=None):
    """Column-wise median imputation. Never a grand mean across features."""
    X = np.asarray(X, dtype=float)
    if medians is None:
        medians = np.nanmedian(X, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(medians, idx[1])
    return X, medians


def fit_ridge(Xtr, ytr, alpha=1.0):
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    return sc, m


def predict(sc, m, X):
    return m.predict(sc.transform(X))


# ------------------------------------- multivariable scheme sensitivity
def fit_sensitivities(opps_tr, channel, alpha_grid=(10, 30, 100, 300)):
    """
    Stage 1: strip out everything that is not the player.
      pts ~ team_season + opponent + game state + condition main effects
    Stage 2: per-player ridge of the residual on the SAME condition matrix.
      L2 shrinks each player's slopes toward the league slope, which is the
      partial-pooling behaviour we want without fitting a full mixed model yet.
    """
    d = opps_tr.filter((pl.col("channel") == channel) & pl.col("charted"))
    if len(d) < 2000:
        return {}, CONDITIONS

    cond = d.select(CONDITIONS).to_numpy().astype(float)
    cond = np.nan_to_num(cond, nan=0.0)

    state = d.select(["down", "ydstogo", "yardline_100", "score_differential",
                      "wp", "half_seconds_remaining"]).to_numpy().astype(float)
    state, _ = prep(state)

    ts = (d["season"].cast(str) + "_" + d["posteam"]).to_list()
    opp = d["defteam"].to_list()
    ts_lv = sorted(set(ts)); opp_lv = sorted(set(opp))
    TS = np.zeros((len(d), len(ts_lv))); OP = np.zeros((len(d), len(opp_lv)))
    TS[np.arange(len(d)), [ts_lv.index(v) for v in ts]] = 1
    OP[np.arange(len(d)), [opp_lv.index(v) for v in opp]] = 1

    y = d["pts"].to_numpy().astype(float)
    X1 = np.hstack([TS, OP, state, cond])
    sc1, m1 = fit_ridge(X1, y, alpha=5.0)
    resid = y - predict(sc1, m1, X1)

    # inner split to pick the shrinkage strength
    pids = d["pid"].to_list()
    order = np.argsort(d["tindex"].to_numpy())
    cut = order[int(len(order) * 0.75):]
    mask_val = np.zeros(len(d), bool); mask_val[cut] = True

    best, best_a = None, alpha_grid[0]
    for a in alpha_grid:
        b = _player_slopes(cond[~mask_val], resid[~mask_val],
                           [p for p, m in zip(pids, mask_val) if not m], a)
        pred = np.array([b.get(p, np.zeros(cond.shape[1])) @ cond[i]
                         for i, p in enumerate(pids) if mask_val[i]])
        err = np.mean((resid[mask_val] - pred) ** 2)
        if best is None or err < best:
            best, best_a = err, a

    slopes = _player_slopes(cond, resid, pids, best_a)
    return slopes, CONDITIONS


def _player_slopes(cond, resid, pids, alpha):
    out = {}
    pids = np.asarray(pids)
    for p in np.unique(pids):
        m = pids == p
        if m.sum() < 25:
            continue
        C = cond[m]
        # ridge closed form, no intercept: residuals are already centred
        A = C.T @ C + alpha * np.eye(C.shape[1])
        out[p] = np.linalg.solve(A, C.T @ resid[m])
    return out


def scheme_term(frame, slopes, conds):
    """Expected per-opportunity scheme effect, in points. Interpretable by
    construction: player's joint slope vector dotted with the offense's
    as-of condition rates."""
    rates = frame.select([f"off_{c}__ewm" for c in conds]).to_numpy().astype(float)
    rates = np.nan_to_num(rates, nan=0.0)
    pid = frame["pid"].to_list()
    k = rates.shape[1]
    return np.array([slopes.get(p, np.zeros(k)) @ rates[i] for i, p in enumerate(pid)])


# ------------------------------------------------------------------ evaluation
def run():
    frame = pl.read_parquet("tables/frame.parquet")
    opps = pl.read_parquet("tables/opportunities.parquet").with_columns(
        (pl.col("season") * 100 + pl.col("week")).alias("tindex"))
    frame = frame.filter(pl.col("p_ngames") >= MIN_HIST)

    rows = []
    for test_season in TEST_SEASONS:
        tr = frame.filter(pl.col("season") < test_season)
        te = frame.filter(pl.col("season") == test_season)
        opps_tr = opps.filter(pl.col("season") < test_season)

        for ch in CHANNELS:
            a = tr.filter(pl.col("channel") == ch)
            b = te.filter(pl.col("channel") == ch)
            if len(a) < 400 or len(b) < 100:
                continue

            # ---- role model
            Xr_tr, med_r = prep(a.select(FEATURES_ROLE).to_numpy())
            Xr_te, _ = prep(b.select(FEATURES_ROLE).to_numpy(), med_r)
            scr, mr = fit_ridge(Xr_tr, a["y_opps"].to_numpy().astype(float), alpha=3.0)
            opps_hat = np.clip(predict(scr, mr, Xr_te), 0, None)

            # ---- efficiency model, baseline
            Xe_tr, med_e = prep(a.select(FEATURES_EFF).to_numpy())
            Xe_te, _ = prep(b.select(FEATURES_EFF).to_numpy(), med_e)
            sce, me = fit_ridge(Xe_tr, a["y_ppo"].to_numpy().astype(float), alpha=3.0)
            ppo_hat = predict(sce, me, Xe_te)

            base_pts = opps_hat * ppo_hat

            # ---- efficiency model, + scheme
            slopes, conds = fit_sensitivities(opps_tr, ch)
            s_tr = scheme_term(a, slopes, conds)
            s_te = scheme_term(b, slopes, conds)

            Xs_tr = np.hstack([a.select(FEATURES_EFF).to_numpy().astype(float),
                               a.select(FEATURES_SCHEME_TEAM).to_numpy().astype(float),
                               s_tr[:, None]])
            Xs_te = np.hstack([b.select(FEATURES_EFF).to_numpy().astype(float),
                               b.select(FEATURES_SCHEME_TEAM).to_numpy().astype(float),
                               s_te[:, None]])
            Xs_tr, med_s = prep(Xs_tr)
            Xs_te, _ = prep(Xs_te, med_s)
            scs, ms = fit_ridge(Xs_tr, a["y_ppo"].to_numpy().astype(float), alpha=3.0)
            ppo_hat_s = predict(scs, ms, Xs_te)
            scheme_pts = opps_hat * ppo_hat_s

            # ---- naive reference: carry forward the player's own scoring rate
            naive = np.nan_to_num(b["pts__ewm"].to_numpy().astype(float) *
                                  0 + b["pts__ewm"].to_numpy().astype(float))

            rows.append(pl.DataFrame({
                "season": [test_season] * len(b), "week": b["week"],
                "channel": [ch] * len(b), "pid": b["pid"], "name": b["name"],
                "pos": b["pos"], "team": b["posteam"],
                "y": b["y_pts"].cast(float),
                "naive": naive, "base": base_pts, "scheme": scheme_pts,
                "scheme_term": s_te, "opps_hat": opps_hat,
                "y_opps": b["y_opps"].cast(float),
            }))

    res = pl.concat(rows)
    res.write_parquet("tables/predictions.parquet")
    print("scored rows:", res.shape)
    return res


if __name__ == "__main__":
    run()
