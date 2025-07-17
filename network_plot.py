#!/usr/bin/env python3
# scripts/network_plot.py  (전체 덮어쓰기)
import pandas as pd, networkx as nx, matplotlib.pyplot as plt
from pathlib import Path

# ─── 경로 ────────────────────────────────────────────────────
bus_loc  = Path("./KPG193_ver1_2/KPG193_ver1_2/network/location/bus_location.csv")
branch   = Path("./KPG193_ver1_2/KPG193_ver1_2/network/branch/branch.csv")
B_csv    = Path("./data/B_41_labeled.csv")
out_png  = Path("./data/KPG_Network_plot.png")

# ─── 데이터 로드 ─────────────────────────────────────────────
bus_df    = pd.read_csv(bus_loc)
branch_df = pd.read_csv(branch)
dfB       = pd.read_csv(B_csv, index_col=0)

def col(df, keys):
    return next(c for c in df.columns if any(k.lower() in c.lower() for k in keys))

idc = col(bus_df, ["bus","id"])
xc  = col(bus_df, ["x","lon"])
yc  = col(bus_df, ["y","lat"])

pos = {row[idc]:(row[xc], row[yc]) for _,row in bus_df.iterrows()}

G_full = nx.Graph()
G_full.add_nodes_from(pos)
G_full.add_edges_from(branch_df[["fbus","tbus"]].values)

gen_buses = dfB.index.astype(int).tolist()
G_gen = nx.Graph()
for i, bi in enumerate(gen_buses):
    for j, bj in enumerate(gen_buses[i+1:], start=i+1):
        w = dfB.iat[i,j]
        if w > 1e-4:
            G_gen.add_edge(bi, bj, weight=w)
max_w = max(nx.get_edge_attributes(G_gen,"weight").values())
edge_w = [d["weight"]/max_w*6 for *_,d in G_gen.edges(data=True)]

# ─── 플롯 ───────────────────────────────────────────────────
fig, ax = plt.subplots(1,2, figsize=(18,9))

# (a) 전체 193-bus
nx.draw(G_full, pos, node_size=18, edge_color="#bdbdbd",
        width=0.3, with_labels=False, ax=ax[0])
ax[0].set_title("KPG-193  (physical lines)")
ax[0].axis("equal")

# (b) 발전기 43-bus
# 1) 연한 회색 노드(발전기 아닌 버스)
non_gen = [b for b in G_full.nodes if b not in gen_buses]
nx.draw_networkx_nodes(G_full, pos, nodelist=non_gen,
                       node_size=12, node_color="#3D3A3A",
                       alpha=0.15, ax=ax[1])

# 2) 연한 회색 물리선 (optional 배경)
nx.draw_networkx_edges(G_full, pos, edgelist=G_full.edges,
                       width=0.2, edge_color="#9A9797",
                       alpha=0.15, ax=ax[1])

# 3) 발전기 간선·노드 강조
nx.draw_networkx_edges(G_gen, pos, width=edge_w,
                       edge_color="royalblue", ax=ax[1])
nx.draw_networkx_nodes(G_gen, pos, nodelist=gen_buses,
                       node_size=50, node_color="orange", ax=ax[1])

ax[1].set_title("41-Generator Effective Network\n(edge width ∝ |B₍ᵢⱼ₎|)")
ax[1].axis("equal")

# ─── 두 축 범위 동일 & 테두리 제거 ───────────────────────────
xlim, ylim = ax[0].get_xlim(), ax[0].get_ylim()
for a in ax:
    a.set_xlim(xlim); a.set_ylim(ylim)
    a.set_axis_off()                   # 스파인·눈금 완전 제거

plt.tight_layout()
out_png.parent.mkdir(exist_ok=True)
plt.savefig(out_png, dpi=300)
print("✅  saved →", out_png)
