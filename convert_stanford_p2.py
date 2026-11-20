import os, sys, json, shutil, argparse, random
import numpy as np
import pandas as pd
from multiprocessing import Pool

REST_MIN = 12
MIN_REST_MIN = 10
GAP_HOURS = 6
BEFORE, AFTER = 7, 13
MIN_PIECE = 30

CFG = {}


def find_files(pid):
    d = os.path.join(CFG["root"], pid)
    for dev, hr, st in (("Fitbit", "Orig_Fitbit_HR.csv", "Orig_Fitbit_ST.csv"),
                        ("AppleWatch", "Orig_NonFitbit_HR.csv", "Orig_NonFitbit_ST.csv")):
        if os.path.isfile(os.path.join(d, hr)):
            return dev, os.path.join(d, hr), os.path.join(d, st)
    return None


def load_hr(path):
    df = pd.read_csv(path, usecols=["datetime", "heartrate"], dtype=str)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["heartrate"] = pd.to_numeric(df["heartrate"], errors="coerce")
    df = df.dropna()
    return df[df.heartrate > 0]


def load_steps(path, device):
    if device == "Fitbit":
        df = pd.read_csv(path, usecols=["datetime", "steps"], dtype=str)
        t0 = pd.to_datetime(df["datetime"], errors="coerce")
        t1 = t0
    else:
        df = pd.read_csv(path, usecols=["start_datetime", "end_datetime", "steps"], dtype=str)
        t0 = pd.to_datetime(df["start_datetime"], errors="coerce")
        t1 = pd.to_datetime(df["end_datetime"], errors="coerce")
    st = pd.to_numeric(df["steps"], errors="coerce")
    ok = t0.notna() & t1.notna() & (st > 0)
    return t0[ok], t1[ok]


def to_min(t):
    return t.values.astype("datetime64[m]").astype("int64")


def resting_hourly(hr, t0s, t1s):
    hr_min = to_min(hr["datetime"])
    lo, hi = hr_min.min(), hr_min.max()
    n = hi - lo + 1
    s0, s1 = to_min(t0s), to_min(t1s)
    s1 = np.maximum(s0, s1)
    keep = (s1 - lo >= 0) & (s0 - lo <= n - 1)
    a = np.clip(s0[keep] - lo, 0, n - 1)
    b = np.clip(s1[keep] - lo, 0, n - 1)
    diff = np.zeros(n + 1, dtype=np.int64)
    np.add.at(diff, a, 1)
    np.add.at(diff, b + 1, -1)
    active = (np.cumsum(diff)[:n] > 0).astype(np.int64)
    ca = np.concatenate([[0], np.cumsum(active)])
    t = np.arange(n)
    steps_before = ca[t] - ca[np.maximum(t - REST_MIN, 0)]
    resting = steps_before == 0
    mask = resting[hr_min - lo]
    tmp = pd.DataFrame({"h": hr_min[mask] // 60, "m": hr_min[mask], "hr": hr["heartrate"].values[mask]})
    g = tmp.groupby("h")
    out = pd.DataFrame({"nmin": g["m"].nunique(), "value": g["hr"].mean()})
    if CFG["count_mode"] == "steps":
        rest_minutes = lo + np.nonzero(resting)[0]
        cnt = pd.Series(rest_minutes // 60).value_counts()
        out["nmin"] = cnt.reindex(out.index).fillna(0).values
    out = out[out.nmin >= MIN_REST_MIN]
    out.index = pd.to_datetime(out.index.values.astype("int64") * 3600, unit="s")
    return out["value"]


def process(pid):
    meta_path = os.path.join(CFG["out"], ".meta", pid + ".json")
    if os.path.exists(meta_path):
        return pid
    f = find_files(pid)
    meta = {"id": pid, "split": CFG["split"].get(pid, "all")}
    lab = CFG["lab"].get(pid)
    meta["covid"] = "yes" if lab is not None else "no"
    meta["type"] = lab["type"] if lab is not None else ""
    meta["day0"] = str(lab["day0"].date()) if lab is not None else ""
    if f is None or not os.path.isfile(f[2]):
        meta.update(device=(lab["device"] if lab is not None else ""), n_pieces=0, n_hours=0,
                    n_window_hours=0, days_monitored=0, note="missing heart rate or steps file")
        json.dump(meta, open(meta_path, "w"))
        return pid
    device, hrp, stp = f
    meta["device"] = device
    s = resting_hourly(load_hr(hrp), *load_steps(stp, device))
    split_dir = os.path.join(CFG["out"], meta["split"])
    os.makedirs(split_dir, exist_ok=True)
    pieces, n_hours, n_win, days = 0, 0, 0, set()
    if len(s):
        gaps = np.diff(s.index.values).astype("timedelta64[s]").astype(np.int64) > GAP_HOURS * 3600
        parts = np.split(np.arange(len(s)), np.where(gaps)[0] + 1)
        for part in parts:
            if len(part) < MIN_PIECE:
                continue
            sub = s.iloc[part]
            dates = sub.index.normalize()
            if lab is not None:
                d0 = lab["day0"].normalize()
                y = ((dates >= d0 - pd.Timedelta(days=BEFORE)) & (dates <= d0 + pd.Timedelta(days=AFTER))).astype(int)
            else:
                y = np.zeros(len(sub), dtype=int)
            pieces += 1
            out = pd.DataFrame({"timestamp": np.arange(1, len(sub) + 1), "value": sub.values,
                                "is_anomaly": np.asarray(y), "datetime": sub.index.strftime("%Y-%m-%d %H:%M:%S")})
            out.to_csv(os.path.join(split_dir, "%s_p%d.csv" % (pid, pieces)), index=False)
            n_hours += len(sub)
            n_win += int(np.sum(y))
            days.update(dates.date)
    meta.update(n_pieces=pieces, n_hours=n_hours, n_window_hours=n_win, days_monitored=len(days), note="")
    json.dump(meta, open(meta_path, "w"))
    return pid


def read_ids(p):
    return [l.strip() for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="datasets/STANFORD_P2_RHR")
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--dev-healthy", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--count-mode", choices=["hr", "steps"], default="steps",
                    help="hr: need 10 resting minutes that have a heart rate reading; steps: need 10 resting minutes by the steps rule")
    a = ap.parse_args()

    lab = pd.read_csv(a.labels, parse_dates=["test_date", "symptom_onset"])
    lab["day0"] = lab.symptom_onset.fillna(lab.test_date)
    CFG["lab"] = {r.id: {"day0": r.day0, "type": r.type, "device": r.device} for r in lab.itertuples()}
    CFG["root"], CFG["out"], CFG["count_mode"] = a.root, a.out, a.count_mode

    if a.train and a.test:
        split = {i: "train" for i in read_ids(a.train)}
        split.update({i: "test" for i in read_ids(a.test)})
        ids = sorted(split)
    else:
        allp = sorted(d for d in os.listdir(a.root) if d.startswith("P") and os.path.isdir(os.path.join(a.root, d)))
        covid = [i for i in allp if i in CFG["lab"]]
        healthy = [i for i in allp if i not in CFG["lab"]]
        random.seed(0); random.shuffle(healthy)
        ids = sorted(covid + healthy[:a.dev_healthy])
        split = {}
    CFG["split"] = split
    os.makedirs(os.path.join(a.out, ".meta"), exist_ok=True)

    todo = [i for i in ids if not os.path.exists(os.path.join(a.out, ".meta", i + ".json"))]
    print("people:", len(ids), "| still to do:", len(todo), flush=True)
    with Pool(a.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(process, todo), 1):
            if k % 10 == 0:
                print("done", k, "of", len(todo), flush=True)

    metas = [json.load(open(os.path.join(a.out, ".meta", i + ".json"))) for i in ids
             if os.path.exists(os.path.join(a.out, ".meta", i + ".json"))]
    if len(metas) < len(ids):
        print("not finished yet, run the same command again (%d of %d people done)" % (len(metas), len(ids)))
        sys.exit(1)
    man = pd.DataFrame(metas)
    man.to_csv(os.path.join(a.out, "manifest.csv"), index=False)

    ndir = os.path.join(a.out, "normal")
    os.makedirs(ndir, exist_ok=True)
    train_name = "train" if a.train else "all"
    for r in man[(man.covid == "no") & (man.split == train_name)].itertuples():
        for fn in os.listdir(os.path.join(a.out, train_name)):
            if fn.startswith(r.id + "_p"):
                shutil.copy(os.path.join(a.out, train_name, fn), os.path.join(ndir, fn))

    print("\nDONE")
    print(man.groupby(["split", "covid"]).agg(people=("id", "count"), pieces=("n_pieces", "sum"),
                                               hours=("n_hours", "sum"), window_hours=("n_window_hours", "sum")))
    print("people with no usable data:", int((man.n_pieces == 0).sum()))


if __name__ == "__main__":
    main()
