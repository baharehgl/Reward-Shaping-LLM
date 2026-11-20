import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BEFORE, AFTER = 7, 13

PAPER = {"covid_detected_symptomatic": "58/66",
         "detected_at_or_before_day0_symptomatic": "53 (of the 58 detected)"}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"


def alerts_for(pr, mode):
    out = []
    for _, g in pr.groupby("piece", sort=True):
        g = g.sort_values("datetime")
        flag = ((g["scored"] == 1) & (g["pred"] == 1)).values
        times, vals = g["datetime"].values, g["value"].values
        prev = False
        for t, v, f in zip(times, vals, flag):
            if f and (mode == "hour" or not prev):
                out.append((pd.Timestamp(t), float(v)))
            prev = f
    return out


def person_row(m, alerts):
    row = {"id": m["id"], "covid": m["covid"], "type": m["type"] if m["covid"] == "yes" else "",
           "device": m["device"], "days_monitored": int(m["days_monitored"]),
           "detected": "", "first_alert_day": np.nan, "false_alerts": 0}
    times = [a[0] for a in alerts]
    if m["covid"] == "yes":
        d0 = pd.Timestamp(m["day0"])
        start, end = d0 - pd.Timedelta(days=BEFORE), d0 + pd.Timedelta(days=AFTER + 1)
        inside = [t for t in times if start <= t < end]
        row["detected"] = "yes" if inside else "no"
        if inside:
            row["first_alert_day"] = (min(inside).normalize() - d0).days
        row["false_alerts"] = len(times) - len(inside)
    else:
        row["false_alerts"] = len(times)
    return row


def prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def point_adjust(df):
    y, p = df["is_anomaly"].values, df["pred"].values.copy()
    for _, idx in df.groupby(["id", "piece"]).indices.items():
        yy, pp = y[idx], p[idx].copy()
        starts = np.where((yy == 1) & (np.r_[0, yy[:-1]] == 0))[0]
        ends = np.where((yy == 1) & (np.r_[yy[1:], 0] == 0))[0]
        for s, e in zip(starts, ends):
            if pp[s:e + 1].any():
                pp[s:e + 1] = 1
        p[idx] = pp
    return p


def summarize(group, res, scored):
    cov = res[res["covid"] == "yes"]
    hea = res[res["covid"] == "no"]
    sym = cov[cov["type"] == "symptomatic"]
    asy = cov[cov["type"] == "asymptomatic"]
    rows = []

    def add(metric, ours, note=""):
        rows.append({"group": group, "metric": metric, "ours": ours,
                     "stanford_paper": PAPER.get(metric, "") if group == "all" else "", "note": note})

    det = lambda d: int((d["detected"] == "yes").sum())
    add("covid_detected_all", "%d/%d" % (det(cov), len(cov)))
    add("covid_detected_symptomatic", "%d/%d" % (det(sym), len(sym)))
    add("covid_detected_asymptomatic", "%d/%d" % (det(asy), len(asy)))
    early = lambda d: int((d["first_alert_day"] <= 0).sum())
    add("detected_at_or_before_day0_all", "%d" % early(cov))
    add("detected_at_or_before_day0_symptomatic", "%d" % early(sym))
    fa = cov["first_alert_day"].dropna()
    add("median_first_alert_day", "%.1f" % fa.median() if len(fa) else "n/a", "detected COVID people")
    fa_s = sym["first_alert_day"].dropna()
    add("median_first_alert_day_symptomatic", "%.1f" % fa_s.median() if len(fa_s) else "n/a")
    days = hea["days_monitored"].sum()
    add("false_alerts_per_healthy_person_per_30_days",
        "%.3f" % (hea["false_alerts"].sum() / days * 30) if days else "n/a",
        "%d healthy people, %d alerts, %d days" % (len(hea), hea["false_alerts"].sum(), days))
    if len(scored):
        y = scored["is_anomaly"].values
        for name, p in (("", scored["pred"].values), ("_point_adjust", point_adjust(scored))):
            pr, rc, f1 = prf(y, p)
            add("precision_hourly" + name, "%.4f" % pr)
            add("recall_hourly" + name, "%.4f" % rc)
            add("f1_hourly" + name, "%.4f" % f1)
    return rows


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def date_axis(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))


def plot_series(ax, g):
    for _, piece in g.groupby("piece", sort=True):
        piece = piece.sort_values("datetime")
        ax.plot(pd.to_datetime(piece["datetime"]), piece["value"], color=BLUE, lw=0.9)


def fig1(preds, res, man, ids, mode, path):
    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 3.1 * len(ids)), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, pid in zip(axes, ids):
        m = man[man["id"] == pid].iloc[0]
        g = preds[preds["id"] == pid]
        d0 = pd.Timestamp(m["day0"])
        style(ax)
        plot_series(ax, g)
        ax.axvspan(d0 - pd.Timedelta(days=BEFORE), d0 + pd.Timedelta(days=AFTER + 1), color=ORANGE, alpha=0.18, lw=0)
        ax.axvline(d0, color=INK, ls="--", lw=1)
        al = alerts_for(g, mode)
        if al:
            ax.scatter([a[0] for a in al], [a[1] for a in al], s=26, color=RED, edgecolor=SURFACE, lw=0.8, zorder=3)
        ax.set_xlim(d0 - pd.Timedelta(days=28), d0 + pd.Timedelta(days=28))
        ax.set_ylabel("Resting HR (bpm)", color=INK2, fontsize=9)
        ax.set_title("%s  |  %s, %s" % (pid, m["device"], m["type"]), loc="left", color=INK, fontsize=10)
        date_axis(ax)
    h = [plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.18), plt.Line2D([], [], color=INK, ls="--"),
         plt.Line2D([], [], color=RED, marker="o", ls="", markersize=5)]
    fig.legend(h, ["illness window (day 0 -7 to +13)", "day 0", "alert"], loc="upper left",
               fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(0.01, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def fig2(res, path):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    style(ax)
    vals = res.loc[(res["covid"] == "yes") & (res["detected"] == "yes"), "first_alert_day"].dropna().astype(int)
    if len(vals):
        ax.hist(vals, bins=np.arange(-BEFORE - 0.5, AFTER + 1.5, 1), color=BLUE, edgecolor=SURFACE, lw=1.5)
        ax.axvline(0, color=INK, ls="--", lw=1)
        ax.text(0.15, ax.get_ylim()[1] * 0.95, "day 0", color=INK2, fontsize=8, va="top")
    else:
        ax.text(0.5, 0.5, "No COVID test person was detected", transform=ax.transAxes, ha="center", color=INK2)
    ax.set_xticks(range(-BEFORE, AFTER + 1))
    ax.set_xlabel("first alert, days from day 0 (negative = before symptoms)", color=INK2, fontsize=9)
    ax.set_ylabel("people", color=INK2, fontsize=9)
    ax.set_title("First alert day, detected COVID test people (n=%d)" % len(vals), loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def fig3(preds, man, pid, mode, path):
    g = preds[preds["id"] == pid]
    m = man[man["id"] == pid].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURFACE)
    style(ax)
    plot_series(ax, g)
    al = alerts_for(g, mode)
    if al:
        ax.scatter([a[0] for a in al], [a[1] for a in al], s=26, color=RED, edgecolor=SURFACE, lw=0.8, zorder=3)
    ax.set_ylabel("Resting HR (bpm)", color=INK2, fontsize=9)
    ax.set_title("%s  |  healthy, %s, %d days monitored, %d alerts" % (pid, m["device"], m["days_monitored"], len(al)),
                 loc="left", color=INK, fontsize=10)
    date_axis(ax)
    ax.legend([plt.Line2D([], [], color=BLUE, lw=1.2), plt.Line2D([], [], color=RED, marker="o", ls="", markersize=5)],
              ["resting heart rate", "alert"], loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/predictions.csv")
    ap.add_argument("--manifest", default="datasets/STANFORD_P2_RHR/manifest.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alert-mode", choices=["run", "hour"], default="run")
    ap.add_argument("--fig1-ids", help="comma-separated COVID test ids for Fig 1 (default: first 3 with data in the window)")
    ap.add_argument("--fig3-id", help="healthy test id for Fig 3 (default: healthy person with most days monitored)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    man = pd.read_csv(a.manifest, dtype=str).fillna("")
    man["days_monitored"] = man["days_monitored"].astype(float).astype(int)
    man["n_window_hours"] = man["n_window_hours"].astype(float).astype(int)
    man = man[man["split"] == "test"].sort_values("id").reset_index(drop=True)
    preds = pd.read_csv(a.pred, dtype={"id": str})
    preds = preds[preds["id"].isin(set(man["id"]))].sort_values(["id", "piece", "datetime"]).reset_index(drop=True)
    print("test people:", len(man), "| with predictions:", preds["id"].nunique())

    rows = [person_row(m, alerts_for(preds[preds["id"] == m["id"]], a.alert_mode)) for _, m in man.iterrows()]
    res = pd.DataFrame(rows)
    cols = ["id", "covid", "type", "device", "days_monitored", "detected", "first_alert_day", "false_alerts"]
    res[cols].to_csv(os.path.join(a.out, "results_per_person.csv"), index=False)

    scored_all = preds[preds["scored"] == 1].reset_index(drop=True)
    summ = []
    for grp in ("all", "Fitbit", "AppleWatch"):
        ids = set(res["id"]) if grp == "all" else set(res.loc[res["device"] == grp, "id"])
        summ += summarize(grp, res[res["id"].isin(ids)], scored_all[scored_all["id"].isin(ids)].reset_index(drop=True))
    pd.DataFrame(summ).to_csv(os.path.join(a.out, "summary.csv"), index=False)

    cov_ok = man[(man["covid"] == "yes") & (man["n_window_hours"] > 0)]["id"].tolist()
    ids1 = a.fig1_ids.split(",") if a.fig1_ids else cov_ok[:3]
    fig1(preds, res, man, ids1, a.alert_mode, os.path.join(a.out, "fig1_covid_resting_hr.png"))
    fig2(res, os.path.join(a.out, "fig2_first_alert_day.png"))
    healthy = man[(man["covid"] == "no") & man["id"].isin(set(preds["id"]))].sort_values("days_monitored", ascending=False)
    id3 = a.fig3_id or healthy["id"].iloc[0]
    fig3(preds, man, id3, a.alert_mode, os.path.join(a.out, "fig3_healthy_example.png"))
    print("Fig 1 people:", ids1, "| Fig 3 person:", id3)
    print("wrote results to", a.out)


if __name__ == "__main__":
    main()
