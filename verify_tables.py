import json, re, os

TOL = 0.002
mismatches = []

def add_mm(table, row, col, paper_val, json_val):
    mismatches.append({
        "table": table, "row": row, "col": col,
        "paper": paper_val, "json": json_val
    })

def close(a, b, tol=TOL):
    try:
        return abs(float(a) - float(b)) <= tol
    except:
        return str(a).strip() == str(b).strip()

# Read LaTeX
with open("paper_scs_framework.tex", "r", encoding="utf-8") as f:
    tex = f.read()

# Load JSONs
with open("results/pipeline_final.json") as f:
    pipeline = json.load(f)
with open("results/experiments/experiment_1_expanded.json") as f:
    exp1 = json.load(f)
with open("results/experiments/power_analysis_summary.json") as f:
    power_json = json.load(f)
with open("results/experiments/experiment_2_corruption.json") as f:
    corr_json = json.load(f)
with open("results/experiments/experiment_oracle_feature.json") as f:
    oracle_json = json.load(f)
with open("results/experiments/fdr_summary.json") as f:
    fdr_json = json.load(f)
with open("results/experiments/experiment_3_threshold.json") as f:
    thresh_json = json.load(f)
with open("results/experiments/multiwindow_oos.json") as f:
    mw_json = json.load(f)

# Group name mapping: paper name -> JSON key
name_map = {
    "3d binary":   "3d_directional_binary",
    "3d ternary":  "3d_multiclass_volatility",
    "5d binary":   "5d_directional_binary",
    "5d ternary":  "5d_multiclass_volatility",
    "7d binary":   "7d_directional_binary",
    "7d ternary":  "7d_multiclass_volatility",
    "10d binary":  "10d_directional_binary",
    "10d ternary": "10d_multiclass_volatility",
    "15d binary":  "15d_directional_binary",
    "15d ternary": "15d_multiclass_volatility",
    "20d binary":  "20d_directional_binary",
    "20d ternary": "20d_multiclass_volatility",
}

def parse_tex_name(raw):
    raw = raw.strip()
    raw = re.sub(r'\\textbf\{([^}]+)\}', r'\1', raw)
    raw = re.sub(r'\\textit\{([^}]+)\}', r'\1', raw)
    raw = raw.strip()
    return raw

def extract_table_block(tex, label):
    pat = re.escape(label)
    m = re.search(pat, tex)
    if not m:
        return None
    start = m.end()
    end_m = re.search(r'\\end\{tabular\}', tex[start:])
    if not end_m:
        return None
    return tex[start:start+end_m.start()]

def parse_number(s):
    s = s.strip()
    s = s.replace('\\$', '$')
    s = re.sub(r'\$[<>]\$', '', s)
    s = s.replace('$-$', '-')
    s = s.replace('$', '')
    s = s.replace('+', '')
    s = s.replace('\\%', '')
    s = s.replace('%', '')
    s = s.replace('\\,', '')
    s = s.replace('{', '').replace('}', '')
    s = s.replace('\\textbf', '')
    s = s.replace('\\checkmark', 'checkmark')
    s = s.strip()
    if s in ('---', '--', '', 'checkmark'):
        return None
    try:
        return float(s)
    except:
        return s

def get_data_lines(block):
    lines = block.split('\n')
    data_lines = []
    in_data = False
    for line in lines:
        if '\\midrule' in line:
            in_data = True
            continue
        if '\\bottomrule' in line:
            break
        if in_data and '&' in line:
            data_lines.append(line)
    return data_lines

# ===== PHASE A =====
print("=" * 70)
print("CHECKING PHASE A (tab:phase_a)")
print("=" * 70)
block = extract_table_block(tex, "tab:phase_a")
phase_a_src = pipeline["phase_a"]["group_results"]

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 8:
        continue
    name_raw = parse_tex_name(parts[0])
    if name_raw not in name_map:
        continue
    jkey = name_map[name_raw]
    if jkey not in phase_a_src:
        print(f"  WARNING: {jkey} not in JSON phase_a")
        continue
    jdata = phase_a_src[jkey]

    checks = [
        ("SCS-A", parse_number(parts[1]), jdata["SCS_A"]),
        ("S_time", parse_number(parts[2]), jdata["S_time"]),
        ("S_asset", parse_number(parts[3]), jdata["S_asset"]),
        ("S_model", parse_number(parts[4]), jdata["S_model"]),
        ("S_seed", parse_number(parts[5]), jdata["S_seed"]),
        ("S_dist", parse_number(parts[6]), jdata["S_dist"]),
    ]
    for col, pv, jv in checks:
        if pv is not None and not close(pv, jv):
            add_mm("Phase A", name_raw, col, pv, jv)
            print(f"  MISMATCH: {name_raw} | {col} | paper={pv} json={jv}")

# ===== PHASE B =====
print("\n" + "=" * 70)
print("CHECKING PHASE B (tab:phase_b)")
print("=" * 70)
block = extract_table_block(tex, "tab:phase_b")
phase_b_src = exp1.get("phase_b", {})
phase_b_pipeline = pipeline["phase_b"]["group_results"]

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 8:
        continue
    name_raw = parse_tex_name(parts[0])
    if name_raw not in name_map:
        continue
    jkey = name_map[name_raw]
    jdata = phase_b_src.get(jkey) or phase_b_pipeline.get(jkey)
    if not jdata:
        print(f"  WARNING: {jkey} not in any phase_b JSON")
        continue

    checks = [
        ("SCS-B", parse_number(parts[1]), jdata["SCS_B"]),
        ("S_time", parse_number(parts[2]), jdata["S_time"]),
        ("S_asset", parse_number(parts[3]), jdata["S_asset"]),
        ("S_cost", parse_number(parts[4]), jdata["S_cost"]),
        ("S_struct", parse_number(parts[5]), jdata["S_struct"]),
        ("S_eco", parse_number(parts[6]), jdata["S_eco"]),
    ]
    for col, pv, jv in checks:
        if pv is not None and not close(pv, jv):
            add_mm("Phase B", name_raw, col, pv, jv)
            print(f"  MISMATCH: {name_raw} | {col} | paper={pv} json={jv}")

# ===== PHASE C =====
print("\n" + "=" * 70)
print("CHECKING PHASE C (tab:phase_c)")
print("=" * 70)
block = extract_table_block(tex, "tab:phase_c")
phase_c_src = exp1.get("phase_c", {})

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 8:
        continue
    if 'SPY' in line or 'B\\&H' in line or 'B&H' in line:
        continue
    name_raw = parse_tex_name(parts[0])
    if name_raw not in name_map:
        continue
    jkey = name_map[name_raw]
    jdata = phase_c_src.get(jkey)
    if not jdata:
        continue

    metrics = jdata["metrics"]
    stats = jdata["statistics"]

    checks = []
    paper_ret = parse_number(parts[1])
    if paper_ret is not None:
        checks.append(("Return", paper_ret, metrics["total_return_pct"]))
    paper_sharpe = parse_number(parts[2])
    if paper_sharpe is not None:
        checks.append(("Sharpe", paper_sharpe, metrics["sharpe_ratio"]))
    paper_maxdd = parse_number(parts[3])
    if paper_maxdd is not None:
        checks.append(("Max DD", paper_maxdd, metrics["max_drawdown_pct"]))
    paper_trades = parse_number(parts[4])
    if paper_trades is not None:
        checks.append(("Trades", paper_trades, metrics["n_trades"]))
    paper_winrate = parse_number(parts[5])
    if paper_winrate is not None:
        checks.append(("Win%", paper_winrate, metrics["win_rate"]))
    paper_dsr_p = parse_number(parts[6])
    if paper_dsr_p is not None:
        checks.append(("DSR p", paper_dsr_p, stats["deflated_sharpe"]["p_value"]))
    paper_bh_p = parse_number(parts[7])
    if paper_bh_p is not None:
        checks.append(("vs B&H p", paper_bh_p, stats["sharpe_test_vs_bh"]["p_value"]))

    for col, pv, jv in checks:
        if not close(pv, jv):
            add_mm("Phase C", name_raw, col, pv, jv)
            print(f"  MISMATCH: {name_raw} | {col} | paper={pv} json={jv}")

# ===== COST TABLE =====
print("\n" + "=" * 70)
print("CHECKING COST TABLE (tab:cost)")
print("=" * 70)
block = extract_table_block(tex, "tab:cost")
cost_levels = ["0", "2", "5", "10", "20", "50"]

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 7:
        continue
    name_raw = parse_tex_name(parts[0])
    if name_raw not in name_map:
        continue
    jkey = name_map[name_raw]
    jdata = phase_c_src.get(jkey) or pipeline["phase_c"]["results"].get(jkey)
    if not jdata:
        continue
    cost_data = jdata["statistics"]["cost_sensitivity"]

    for i, cl in enumerate(cost_levels):
        paper_val = parse_number(parts[i + 1])
        if paper_val is None:
            continue
        json_val = cost_data[cl]["sharpe"]
        if not close(paper_val, json_val):
            add_mm("Cost", name_raw, f"{cl}bps", paper_val, json_val)
            print(f"  MISMATCH: {name_raw} | {cl}bps | paper={paper_val} json={json_val}")

# ===== POWER TABLE =====
print("\n" + "=" * 70)
print("CHECKING POWER TABLE (tab:power)")
print("=" * 70)
block = extract_table_block(tex, "tab:power")

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 7:
        continue
    k_str = parse_number(parts[0])
    if k_str is None:
        continue
    k_float = float(k_str)
    k_key = f"k_{k_float}"

    jdata = power_json.get(k_key)
    if not jdata:
        print(f"  WARNING: k={k_str} not in power JSON (key={k_key})")
        continue

    checks = [
        ("Mean SCS-A", parse_number(parts[1]), jdata["mean_scs_a"]),
        ("Std", parse_number(parts[2]), jdata["std_scs_a"]),
        ("IS SR", parse_number(parts[3]), jdata["mean_sharpe"]),
    ]

    paper_p70 = parse_number(parts[4])
    if paper_p70 is not None:
        json_p70 = jdata["power"]["tau_0.70"] * 100
        if not close(paper_p70, json_p70):
            checks.append(("Power@0.70", paper_p70, json_p70))
    paper_p75 = parse_number(parts[5])
    if paper_p75 is not None:
        json_p75 = jdata["power"]["tau_0.75"] * 100
        if not close(paper_p75, json_p75):
            checks.append(("Power@0.75", paper_p75, json_p75))
    paper_p80 = parse_number(parts[6])
    if paper_p80 is not None:
        json_p80 = jdata["power"]["tau_0.80"] * 100
        if not close(paper_p80, json_p80):
            checks.append(("Power@0.80", paper_p80, json_p80))

    for col, pv, jv in checks:
        if pv is not None and not close(pv, jv):
            add_mm("Power", f"k={k_str}", col, pv, jv)
            print(f"  MISMATCH: k={k_str} | {col} | paper={pv} json={jv}")

# ===== CORRUPTION TABLE =====
print("\n" + "=" * 70)
print("CHECKING CORRUPTION TABLE (tab:corruption)")
print("=" * 70)
block = extract_table_block(tex, "tab:corruption")
corruption_levels = ["0", "10", "20", "30", "50", "75", "100"]

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 8:
        continue
    metric_name = parts[0].strip()

    if "Mean SCS-A" in metric_name:
        for i, cl in enumerate(corruption_levels):
            paper_val = parse_number(parts[i + 1])
            if paper_val is None:
                continue
            corr_data = corr_json.get(cl)
            if not corr_data:
                continue
            all_scs = [corr_data[g]["SCS_A"] for g in corr_data]
            json_mean = sum(all_scs) / len(all_scs)
            if not close(paper_val, json_mean, tol=0.002):
                add_mm("Corruption", "Mean SCS-A", f"{cl}%", paper_val, round(json_mean, 3))
                print(f"  MISMATCH: Mean SCS-A | {cl}% | paper={paper_val} json={round(json_mean, 3)}")

    elif "N" in metric_name and "pass" in metric_name:
        for i, cl in enumerate(corruption_levels):
            paper_val = parse_number(parts[i + 1])
            if paper_val is None:
                continue
            corr_data = corr_json.get(cl)
            if not corr_data:
                continue
            n_pass = sum(1 for g in corr_data if corr_data[g]["SCS_A"] >= 0.70)
            if not close(paper_val, n_pass):
                add_mm("Corruption", "N_pass", f"{cl}%", paper_val, n_pass)
                print(f"  MISMATCH: N_pass | {cl}% | paper={paper_val} json={n_pass}")

# ===== ORACLE TABLE =====
print("\n" + "=" * 70)
print("CHECKING ORACLE TABLE (tab:synthetic)")
print("=" * 70)
block = extract_table_block(tex, "tab:synthetic")
oracle_calib = oracle_json["calibration"]
oracle_dose = oracle_json["dose_response"]

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 11:
        continue

    k_val = parse_number(parts[0])
    if k_val is None:
        continue
    k_key = f"k={k_val}"

    calib = oracle_calib.get(k_key)
    dose = oracle_dose.get(k_key)
    if not calib or not dose:
        print(f"  WARNING: {k_key} not in oracle JSON")
        continue

    checks = [
        ("Corr", parse_number(parts[1]), round(calib["corr"], 3)),
        ("Acc", parse_number(parts[2]), round(calib["accuracy"] * 100, 1)),
        ("S_time", parse_number(parts[3]), dose["S_time"]),
        ("S_asset", parse_number(parts[4]), dose["S_asset"]),
        ("S_model", parse_number(parts[5]), dose["S_model"]),
        ("S_seed", parse_number(parts[6]), dose["S_seed"]),
        ("S_dist", parse_number(parts[7]), dose["S_dist"]),
        ("SCS-A", parse_number(parts[8]), dose["SCS_A"]),
        ("IS SR", parse_number(parts[10]), dose["mean_sharpe"]),
    ]

    for col, pv, jv in checks:
        if pv is not None and not close(pv, jv):
            add_mm("Oracle", f"k={k_val}", col, pv, jv)
            print(f"  MISMATCH: k={k_val} | {col} | paper={pv} json={jv}")

# ===== FDR TABLE =====
print("\n" + "=" * 70)
print("CHECKING FDR TABLE (tab:fdr)")
print("=" * 70)
block = extract_table_block(tex, "tab:fdr")

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 5:
        continue

    tau_val = parse_number(parts[0])
    if tau_val is None:
        continue
    tau_key = f"tau_{tau_val:.2f}"

    jdata = fdr_json.get(tau_key)
    if not jdata:
        print(f"  WARNING: tau={tau_val} not in FDR JSON (key={tau_key})")
        continue

    paper_mean_n = parse_number(parts[1])
    ci_raw = parts[2].strip().replace('[', '').replace(']', '').replace('\\,', '')
    ci_parts = ci_raw.split(',')
    paper_ci_lo = parse_number(ci_parts[0]) if len(ci_parts) >= 2 else None
    paper_ci_hi = parse_number(ci_parts[1]) if len(ci_parts) >= 2 else None

    paper_fpr_bin = parse_number(parts[3])
    paper_fpr_tern = parse_number(parts[4])

    json_fpr_bin = jdata["mean_fpr_binary"] * 100
    json_fpr_tern = jdata["mean_fpr_ternary"] * 100

    checks = [
        ("Mean N_pass", paper_mean_n, jdata["mean_n_pass"]),
        ("CI lo", paper_ci_lo, jdata["ci_95_lo"]),
        ("CI hi", paper_ci_hi, jdata["ci_95_hi"]),
        ("FPR binary", paper_fpr_bin, json_fpr_bin),
        ("FPR ternary", paper_fpr_tern, json_fpr_tern),
    ]

    for col, pv, jv in checks:
        if pv is not None and not close(pv, jv, tol=0.02):
            add_mm("FDR", f"tau={tau_val}", col, pv, jv)
            print(f"  MISMATCH: tau={tau_val} | {col} | paper={pv} json={jv}")

# ===== THRESHOLD TABLE =====
print("\n" + "=" * 70)
print("CHECKING THRESHOLD TABLE (tab:threshold)")
print("=" * 70)
block = extract_table_block(tex, "tab:threshold")
thresh_table = thresh_json["threshold_table"]
thresh_lookup = {t["threshold"]: t for t in thresh_table}

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 5:
        continue

    tau_val = parse_number(parts[0])
    if tau_val is None:
        continue

    jdata = thresh_lookup.get(tau_val)
    if not jdata:
        continue

    paper_npass = parse_number(parts[1])
    paper_sharpe = parse_number(parts[2])
    paper_ret = parse_number(parts[3])

    if paper_npass is not None and not close(paper_npass, jdata["n_passing"]):
        add_mm("Threshold", f"tau={tau_val}", "N_pass", paper_npass, jdata["n_passing"])
        print(f"  MISMATCH: tau={tau_val} | N_pass | paper={paper_npass} json={jdata['n_passing']}")

    if paper_sharpe is not None and jdata["avg_oos_sharpe"] is not None:
        if not close(paper_sharpe, jdata["avg_oos_sharpe"]):
            add_mm("Threshold", f"tau={tau_val}", "Avg Sharpe", paper_sharpe, jdata["avg_oos_sharpe"])
            print(f"  MISMATCH: tau={tau_val} | Avg Sharpe | paper={paper_sharpe} json={jdata['avg_oos_sharpe']}")

    if paper_ret is not None and jdata["avg_oos_return"] is not None:
        if not close(paper_ret, jdata["avg_oos_return"]):
            add_mm("Threshold", f"tau={tau_val}", "Avg Return", paper_ret, jdata["avg_oos_return"])
            print(f"  MISMATCH: tau={tau_val} | Avg Return | paper={paper_ret} json={jdata['avg_oos_return']}")

# ===== WEIGHT SENSITIVITY TABLE =====
print("\n" + "=" * 70)
print("CHECKING WEIGHT SENSITIVITY TABLE (tab:weight_sensitivity)")
print("=" * 70)
block = extract_table_block(tex, "tab:weight_sensitivity")
all_scs_a = [phase_a_src[g]["SCS_A"] for g in phase_a_src]
json_n_pass = sum(1 for s in all_scs_a if s >= 0.70)
json_min_scs = min(all_scs_a)
json_mean_scs = sum(all_scs_a) / len(all_scs_a)

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 5:
        continue
    config = parts[0].strip()
    if "baseline" in config.lower() or "W0" in config:
        paper_npass = parse_number(parts[2])
        paper_min = parse_number(parts[3])
        paper_mean = parse_number(parts[4])

        if paper_npass is not None and not close(paper_npass, json_n_pass):
            add_mm("Weight Sensitivity", "W0", "N_pass", paper_npass, json_n_pass)
            print(f"  MISMATCH: W0 | N_pass | paper={paper_npass} json={json_n_pass}")
        if paper_min is not None and not close(paper_min, json_min_scs):
            add_mm("Weight Sensitivity", "W0", "Min SCS-A", paper_min, json_min_scs)
            print(f"  MISMATCH: W0 | Min SCS-A | paper={paper_min} json={json_min_scs}")
        if paper_mean is not None and not close(paper_mean, json_mean_scs):
            add_mm("Weight Sensitivity", "W0", "Mean SCS-A", paper_mean, round(json_mean_scs, 3))
            print(f"  MISMATCH: W0 | Mean SCS-A | paper={paper_mean} json={round(json_mean_scs, 3)}")

# ===== MULTIWINDOW TABLE =====
print("\n" + "=" * 70)
print("CHECKING MULTIWINDOW TABLE (tab:multiwindow_results)")
print("=" * 70)
block = extract_table_block(tex, "tab:multiwindow_results")

mw_sharpe = {}
mw_scsb = {}
for year in ["2023", "2024", "2025"]:
    mw_sharpe[year] = {}
    mw_scsb[year] = {}
    if year in mw_json:
        if "phase_c" in mw_json[year]:
            for gkey, gdata in mw_json[year]["phase_c"].items():
                mw_sharpe[year][gkey] = gdata["metrics"]["sharpe_ratio"]
        if "phase_b" in mw_json[year]:
            for gkey, gdata in mw_json[year]["phase_b"].items():
                mw_scsb[year][gkey] = gdata["SCS_B"]

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 5:
        continue

    name_raw = parse_tex_name(parts[0])
    if 'Avg' in name_raw or 'avg' in name_raw:
        continue
    if name_raw not in name_map:
        continue
    jkey = name_map[name_raw]

    windows = [("2023", 1), ("2024", 2), ("2025", 3)]
    for wi, (year, wnum) in enumerate(windows):
        paper_sr = parse_number(parts[wi + 1])
        if paper_sr is None:
            continue

        json_sr = mw_sharpe.get(year, {}).get(jkey)
        if json_sr is None:
            add_mm("Multiwindow", name_raw, f"W{wnum} SR", paper_sr, "NOT IN JSON")
            print(f"  MISMATCH: {name_raw} | W{wnum} SR | paper={paper_sr} json=NOT IN JSON")
            continue

        if not close(paper_sr, json_sr):
            add_mm("Multiwindow", name_raw, f"W{wnum} SR", paper_sr, json_sr)
            print(f"  MISMATCH: {name_raw} | W{wnum} SR | paper={paper_sr} json={json_sr}")

# ===== WINDOW CONTRAST TABLE =====
print("\n" + "=" * 70)
print("CHECKING WINDOW CONTRAST TABLE (tab:window_contrast)")
print("=" * 70)
block = extract_table_block(tex, "tab:window_contrast")

for line in get_data_lines(block):
    line = line.replace('\\\\', '').strip()
    parts = [p.strip() for p in line.split('&')]
    if len(parts) < 5:
        continue
    window_name = parts[0].strip()
    if "2010" in window_name:
        all_scs = [phase_a_src[g]["SCS_A"] for g in phase_a_src]
        json_npass = sum(1 for s in all_scs if s >= 0.70)
        json_mean = sum(all_scs) / len(all_scs)
        json_min = min(all_scs)

        npass_raw = parts[1].strip()
        npass_match = re.match(r'(\d+)/12', npass_raw)
        if npass_match:
            paper_npass_int = int(npass_match.group(1))
            if paper_npass_int != json_npass:
                add_mm("Window Contrast", "2010-2013", "N_pass", paper_npass_int, json_npass)
                print(f"  MISMATCH: 2010-2013 | N_pass | paper={paper_npass_int} json={json_npass}")

        paper_mean = parse_number(parts[2])
        if paper_mean is not None and not close(paper_mean, json_mean, tol=0.01):
            add_mm("Window Contrast", "2010-2013", "Mean SCS-A", paper_mean, round(json_mean, 2))
            print(f"  MISMATCH: 2010-2013 | Mean SCS-A | paper={paper_mean} json={round(json_mean, 2)}")

        paper_min = parse_number(parts[3])
        if paper_min is not None and not close(paper_min, json_min):
            add_mm("Window Contrast", "2010-2013", "Min SCS-A", paper_min, json_min)
            print(f"  MISMATCH: 2010-2013 | Min SCS-A | paper={paper_min} json={json_min}")

# ===== BOOTSTRAP TABLE =====
print("\n" + "=" * 70)
print("CHECKING BOOTSTRAP TABLE (tab:bootstrap)")
print("=" * 70)
block = extract_table_block(tex, "tab:bootstrap")
if block:
    boot_src = exp1["phase_c"]["20d_directional_binary"]["statistics"]["bootstrap"]

    for line in get_data_lines(block):
        line = line.replace('\\\\', '').strip()
        parts = [p.strip() for p in line.split('&')]
        if len(parts) < 4:
            continue
        metric_name = parts[0].strip()
        paper_point = parse_number(parts[1])
        paper_lo = parse_number(parts[2])
        paper_hi = parse_number(parts[3])

        if "Sharpe" in metric_name:
            jd = boot_src["sharpe_daily"]
            checks = [
                ("Sharpe point", paper_point, jd["point"]),
                ("Sharpe CI lo", paper_lo, jd["ci_lower"]),
                ("Sharpe CI hi", paper_hi, jd["ci_upper"]),
            ]
        elif "return" in metric_name.lower():
            jd = boot_src["total_return"]
            checks = [
                ("Return point", paper_point, jd["point"]),
                ("Return CI lo", paper_lo, jd["ci_lower"]),
                ("Return CI hi", paper_hi, jd["ci_upper"]),
            ]
        elif "Win" in metric_name or "win" in metric_name:
            jd = boot_src["win_rate"]
            checks = [
                ("WinRate point", paper_point, jd["point"]),
                ("WinRate CI lo", paper_lo, jd["ci_lower"]),
                ("WinRate CI hi", paper_hi, jd["ci_upper"]),
            ]
        else:
            continue

        for col, pv, jv in checks:
            if pv is not None and not close(pv, jv, tol=0.05):
                add_mm("Bootstrap", "20d binary", col, pv, jv)
                print(f"  MISMATCH: 20d binary | {col} | paper={pv} json={jv}")

# ===== COMPARISON TABLE =====
print("\n" + "=" * 70)
print("CHECKING COMPARISON TABLE (tab:comparison)")
print("=" * 70)
block = extract_table_block(tex, "tab:comparison")
if block:
    for line in get_data_lines(block):
        line = line.replace('\\\\', '').strip()
        parts = [p.strip() for p in line.split('&')]
        if len(parts) < 5:
            continue
        strat = parts[0].strip()
        paper_n = parse_number(parts[1])
        paper_avg_sr = parse_number(parts[2])

        if "All 12" in strat or "no filter" in strat:
            all_sharpes = [exp1["phase_c"][g]["metrics"]["sharpe_ratio"] for g in exp1["phase_c"]]
            json_avg = sum(all_sharpes) / len(all_sharpes)
            if paper_avg_sr is not None and not close(paper_avg_sr, json_avg):
                add_mm("Comparison", "All 12", "Avg OOS SR", paper_avg_sr, round(json_avg, 4))
                print(f"  MISMATCH: All 12 | Avg OOS SR | paper={paper_avg_sr} json={round(json_avg, 4)}")

        elif "0.70" in strat and "SCS" in strat:
            approved_keys = [name_map[n] for n in name_map if pipeline["phase_a"]["group_results"][name_map[n]]["SCS_A"] >= 0.70]
            approved_sharpes = [exp1["phase_c"][g]["metrics"]["sharpe_ratio"] for g in approved_keys if g in exp1["phase_c"]]
            json_avg = sum(approved_sharpes) / len(approved_sharpes)
            if paper_avg_sr is not None and not close(paper_avg_sr, json_avg):
                add_mm("Comparison", "SCS 0.70", "Avg OOS SR", paper_avg_sr, round(json_avg, 4))
                print(f"  MISMATCH: SCS 0.70 | Avg OOS SR | paper={paper_avg_sr} json={round(json_avg, 4)}")

# ===== FINAL SUMMARY =====
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

if len(mismatches) == 0:
    print("\n*** ALL TABLES MATCH ***")
else:
    print(f"\n*** FOUND {len(mismatches)} MISMATCH(ES) ***\n")
    for mm in mismatches:
        print(f"  Table: {mm['table']:<20} Row: {mm['row']:<20} Col: {mm['col']:<15} Paper: {str(mm['paper']):<12} JSON: {mm['json']}")
