#!/usr/bin/env python3
# make_B.py  ―  MATPOWER .m → Ybus → Kron → B (labeled, active generators only)

import re, numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix

# ─── 경로 ────────────────────────────────────────────────────
CASE = Path("./KPG193_ver1_2/KPG193_ver1_2/network/m/KPG193_ver1_2.m")
DATA = Path("./data"); DATA.mkdir(exist_ok=True)

# ─── 0. .m 파서 유틸 ─────────────────────────────────────────
mtext = CASE.read_text(encoding="utf-8", errors="ignore")
num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
def block(key):
    m = re.search(rf"(?:mpc\.)?{key}\s*=\s*\[([\s\S]+?)\];", mtext, re.I)
    rows=[];  txt = m.group(1).splitlines()
    for ln in txt:
        ln = ln.split("%",1)[0].strip().rstrip(";")
        if ln: rows.append([float(x) for x in re.findall(num, ln)])
    n = max(map(len,rows)); arr = np.zeros((len(rows), n))
    for i,r in enumerate(rows): arr[i,:len(r)] = r
    return arr
def scalar(key):
    return float(re.search(rf"{key}\s*=\s*({num})", mtext).group(1))

# ─── 1. Ybus 원 계산 ─────────────────────────────────────────
baseMVA = scalar("baseMVA")
bus, branch = block("bus"), block("branch")
bus_id = bus[:,0].astype(int)
id2i = {b:i for i,b in enumerate(bus_id)}

rows=cols=vals=[]
rows=[]; cols=[]; vals=[]
for br in branch:
    if int(br[10])!=1: continue
    f,t = map(int, br[:2]); r,x,b = br[2:5]
    if r==0 and x==0: continue
    y  = 1/complex(r,x)
    tap= br[8] if br[8] else 1.0
    tap*= np.exp(1j*br[9]*np.pi/180)
    yff = (y+1j*b/2)/(tap*np.conj(tap)); ytt = y+1j*b/2
    yft = -y/np.conj(tap);               ytf = -y/tap
    i,j = id2i[f], id2i[t]
    rows += [i,i,j,j]; cols += [i,j,i,j]; vals += [yff,yft,ytf,ytt]

Gs, Bs = bus[:,4]/baseMVA, bus[:,5]/baseMVA
for i,(g,s) in enumerate(zip(Gs,Bs)):
    if g or s: rows.append(i); cols.append(i); vals.append(complex(g,s))

Ybus = csr_matrix((vals,(rows,cols)), shape=(len(bus),)*2).toarray()

# ─── 2. 섬 4버스 제거 → 189 ─────────────────────────────────
islands = {119,164,185,186}
mask = np.array([b not in islands for b in bus_id])
bus_id_red = bus_id[mask]
Yred = Ybus[np.ix_(mask,mask)]

# ─── 3. 가동 발전기(STATUS==1)만 선택 → Kron 축소 ──────────
gen      = block("gen")
gen_buses= np.unique(gen[gen[:,7]==1, 0].astype(int))   # 중복 제거!
gen_mask = np.isin(bus_id_red, gen_buses)
load_mask= ~gen_mask

Ygg = Yred[np.ix_(gen_mask, gen_mask)]
Ygl = Yred[np.ix_(gen_mask, load_mask)]
Yll = Yred[np.ix_(load_mask, load_mask)]
Ygen= Ygg - Ygl @ np.linalg.inv(Yll) @ Ygl.T

# ─── 4. B 라플라시안 ────────────────────────────────────────
B = -Ygen.imag
np.fill_diagonal(B, -(B.sum(1)-np.diag(B)))   # 행합 0
B *= -1                                       # 대각 -, 오프 +

n = B.shape[0]
print(f"► B shape = {n} × {n}")               # e.g. 41×41

# ─── 5. 저장: 숫자 CSV + 헤더 CSV + npy 메타 ────────────────
raw_csv = DATA / f"B_{n}.csv"
lbl_csv = DATA / f"B_{n}_labeled.csv"
np.savetxt(raw_csv, B, delimiter=",", fmt="%.12g")
pd.DataFrame(B, index=gen_buses, columns=gen_buses
            ).to_csv(lbl_csv, float_format="%.12g")
np.save(DATA/"bus_id_red.npy", bus_id_red)
np.save(DATA/"gen_mask_active.npy", gen_mask)

print(f"✅  saved → {raw_csv.name} & {lbl_csv.name}")
