import pandas as pd
import networkx as nx

bus_location_path = r"./KPG193_ver1_2/KPG193_ver1_2/network/location/bus_location.csv"
branch_path = r"./KPG193_ver1_2/KPG193_ver1_2/network/branch/branch.csv"
gen_path = r"./data/gen_result.csv"

# Load data
bus_df = pd.read_csv(bus_location_path)
branch_df = pd.read_csv(branch_path)
gen_df = pd.read_csv(gen_path)

# 1. bus_id에서 결측치 행 제거
bus_df = bus_df.dropna(subset=['bus_id'])

# 2. bus_id, fbus, tbus, gen_df['bus'] 모두 int로 강제 변환
bus_df['bus_id'] = bus_df['bus_id'].astype(int)
branch_df['fbus'] = branch_df['fbus'].astype(int)
branch_df['tbus'] = branch_df['tbus'].astype(int)
if 'bus' in gen_df.columns:
    gen_df['bus'] = gen_df['bus'].astype(int)
if 'gen_id' in gen_df.columns:
    gen_df['gen_id'] = gen_df['gen_id'].astype(int)

def find_col(df, key_candidates):
    for key in df.columns:
        for cand in key_candidates:
            if cand.lower() in key.lower():
                return key
    raise ValueError(f"No column found for keys: {key_candidates}")

bus_id_col = find_col(bus_df, ["bus_id", "bus", "id", "번호"])
loc_col = find_col(bus_df, ["location", "지명", "loc", "name"])

bus_to_gen = dict()
for _, row in gen_df.iterrows():
    bus = row['bus']
    gen_id = row['gen_id']
    if bus not in bus_to_gen:
        bus_to_gen[bus] = []
    bus_to_gen[bus].append(str(gen_id))

# 모든 버스 노드 추가(고립노드 포함)
G = nx.Graph()
G.add_nodes_from(bus_df[bus_id_col])
for _, row in branch_df.iterrows():
    G.add_edge(row['fbus'], row['tbus'])

records = []
for _, row in bus_df.iterrows():
    bus_id = int(row[bus_id_col])
    gen_ids = ",".join(bus_to_gen.get(bus_id, []))
    location = row[loc_col]
    if bus_id in G:
        neighbors = list(G.neighbors(bus_id))
    else:
        neighbors = []
    neighbor_str = ",".join(map(str, neighbors)) if neighbors else ""
    neighbor_gens = []
    for n in neighbors:
        neighbor_gens.extend(bus_to_gen.get(n, []))
    neighbor_gen_str = ",".join(neighbor_gens) if neighbor_gens else ""
    records.append([bus_id, gen_ids, location, neighbor_str, neighbor_gen_str])

summary_df = pd.DataFrame(records, columns=[
    "bus_id",
    "generator_ids (at this bus)",
    "location_name",
    "neighbor_buses",
    "neighbor_generator_ids"
])
summary_df.to_csv("./data/node_summary.csv", index=False)

print("node_summary.csv successfully created.")
