"""
Stage 6: diagnostics.

7. Team-change natural experiment. Players who changed teams between seasons:
   does the predicted scheme effect line up with the observed change in
   points per opportunity?

8. Persistence. A feature that does not repeat itself cannot forecast.
   Measured for team tendencies AND for the player sensitivity slopes, which
   is the term the ablation found to be actively harmful.
"""
import numpy as np, polars as pl
from scipy import stats
from model import fit_sensitivities, CHANNELS
from features import CONDITIONS

opps = pl.read_parquet("tables/opportunities.parquet").with_columns(
    (pl.col("season") * 100 + pl.col("week")).alias("tindex"))
pg = pl.read_parquet("tables/player_game.parquet")

SEASONS = [2022, 2023, 2024, 2025]


# ----------------------------------------------------------- 8a team tendencies
def team_persistence():
    ch = opps.filter(pl.col("charted"))
    t = ch.group_by(["season", "posteam"]).agg(
        [pl.col(c).mean().alias(c) for c in CONDITIONS] +
        [pl.col("n_defense_box").mean().alias("box_faced"), pl.len().alias("n")]
    ).filter(pl.col("n") > 300)

    print("8a. TEAM TENDENCY PERSISTENCE, season t vs t+1")
    print(f"    {'tendency':20s} {'r':>7s} {'n pairs':>8s}")
    feats = CONDITIONS + ["box_faced"]
    for f in feats:
        xs, ys = [], []
        for s in SEASONS[:-1]:
            a = t.filter(pl.col("season") == s).select(["posteam", f])
            b = t.filter(pl.col("season") == s + 1).select(["posteam", f]).rename({f: "nxt"})
            j = a.join(b, on="posteam")
            xs += j[f].to_list(); ys += j["nxt"].to_list()
        r = stats.pearsonr(xs, ys).statistic
        print(f"    {f:20s} {r:7.3f} {len(xs):8d}")


# ----------------------------------------------- 8b player sensitivity slopes
def sensitivity_persistence():
    print("\n8b. PLAYER SENSITIVITY PERSISTENCE")
    print("    slopes fit on disjoint season pairs, then correlated per condition")
    print(f"    {'channel':10s} {'players':>8s} {'mean r':>8s} {'median r':>9s}")
    for ch in CHANNELS:
        early = opps.filter(pl.col("season").is_in([2022, 2023]))
        late = opps.filter(pl.col("season").is_in([2024, 2025]))
        sa, conds = fit_sensitivities(early, ch)
        sb, _ = fit_sensitivities(late, ch)
        common = sorted(set(sa) & set(sb))
        if len(common) < 15:
            print(f"    {ch:10s} {len(common):8d}      too few")
            continue
        A = np.array([sa[p] for p in common]); B = np.array([sb[p] for p in common])
        rs = [stats.pearsonr(A[:, j], B[:, j]).statistic for j in range(A.shape[1])]
        print(f"    {ch:10s} {len(common):8d} {np.mean(rs):8.3f} {np.median(rs):9.3f}")
        if ch == "wrte_rec":
            print("      by condition:")
            for j, c in enumerate(conds):
                print(f"        {c:20s} r={rs[j]:+.3f}")


# ----------------------------------------------- 7 team-change experiment
def team_change():
    print("\n7. TEAM-CHANGE NATURAL EXPERIMENT")
    # season-level ppo per player-channel, plus team played for
    ps = (pg.group_by(["season", "pid", "name", "channel", "posteam"])
            .agg([pl.col("opps").sum().alias("opps"), pl.col("pts").sum().alias("pts")])
            .filter(pl.col("opps") >= 40)
            .with_columns((pl.col("pts") / pl.col("opps")).alias("ppo")))
    # keep the team where the player had the most volume that season
    ps = ps.sort("opps", descending=True).unique(
        subset=["season", "pid", "channel"], keep="first")

    ch_scheme = opps.filter(pl.col("charted")).group_by(["season", "posteam"]).agg(
        [pl.col(c).mean().alias(f"r_{c}") for c in CONDITIONS])

    rows = []
    for ch in CHANNELS:
        slopes, conds = fit_sensitivities(
            opps.filter(pl.col("season").is_in([2022, 2023])), ch)
        if not slopes:
            continue
        d = ps.filter(pl.col("channel") == ch)
        for s in [2023, 2024]:
            a = d.filter(pl.col("season") == s)
            b = d.filter(pl.col("season") == s + 1).select(
                ["pid", "posteam", "ppo", "opps"]).rename(
                {"posteam": "team_b", "ppo": "ppo_b", "opps": "opps_b"})
            j = a.join(b, on="pid").filter(pl.col("posteam") != pl.col("team_b"))
            if not len(j):
                continue
            ja = j.join(ch_scheme.filter(pl.col("season") == s).drop("season"),
                        on="posteam", how="left")
            jb = ch_scheme.filter(pl.col("season") == s + 1).drop("season").rename(
                {f"r_{c}": f"n_{c}" for c in conds})
            ja = ja.join(jb, left_on="team_b", right_on="posteam", how="left")
            for r in ja.iter_rows(named=True):
                if r["pid"] not in slopes:
                    continue
                beta = slopes[r["pid"]]
                old = np.array([r.get(f"r_{c}") or 0 for c in conds])
                new = np.array([r.get(f"n_{c}") or 0 for c in conds])
                rows.append({"channel": ch, "name": r["name"],
                             "pred_delta": float(beta @ (new - old)),
                             "actual_delta": r["ppo_b"] - r["ppo"],
                             "opps": min(r["opps"], r["opps_b"])})
    D = pl.DataFrame(rows)
    print(f"    movers with enough volume on both sides: {len(D)}")
    if len(D) > 20:
        r = stats.pearsonr(D["pred_delta"], D["actual_delta"])
        sp = stats.spearmanr(D["pred_delta"], D["actual_delta"])
        print(f"    predicted vs actual change in points per opportunity")
        print(f"      pearson  r={r.statistic:+.3f}  p={r.pvalue:.3f}")
        print(f"      spearman r={sp.statistic:+.3f}  p={sp.pvalue:.3f}")
        sl = stats.linregress(D["pred_delta"], D["actual_delta"])
        print(f"      slope={sl.slope:+.3f} (1.0 would mean correctly scaled)")
        for ch in D["channel"].unique().to_list():
            s = D.filter(pl.col("channel") == ch)
            if len(s) >= 15:
                rr = stats.pearsonr(s["pred_delta"], s["actual_delta"]).statistic
                print(f"      {ch:10s} n={len(s):3d}  r={rr:+.3f}")
    D.write_parquet("tables/team_change.parquet")


if __name__ == "__main__":
    team_persistence()
    sensitivity_persistence()
    team_change()
