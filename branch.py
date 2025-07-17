import re
import pandas as pd

matpower_path = "./KPG193_ver1_2/KPG193_ver1_2/network/m/KPG193_ver1_2.m"
out_csv_path = "./KPG193_ver1_2/KPG193_ver1_2/network/branch/branch.csv"

with open(matpower_path, encoding="utf-8") as f:
    lines = f.readlines()

inside_branch = False
branch_lines = []
for line in lines:
    if "mpc.branch" in line and "=" in line:
        inside_branch = True
        continue
    if inside_branch and "];" in line:
        inside_branch = False
        break
    if inside_branch:
        l = line.strip()
        if l and not l.startswith("%"):
            branch_lines.append(l)

rows = []
for l in branch_lines:
    l = re.sub(r'%.*', '', l)
    vals = [x.replace(';', '') for x in l.split()]
    vals = [float(x) for x in vals if x]
    if len(vals) >= 2:
        rows.append([int(vals[0]), int(vals[1])])

branch_df = pd.DataFrame(rows, columns=["fbus", "tbus"])
branch_df.to_csv(out_csv_path, index=False)
print(f"Saved: {out_csv_path}")
