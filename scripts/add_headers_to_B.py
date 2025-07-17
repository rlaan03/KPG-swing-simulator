# ─────────────────── scripts/add_headers_to_B.py  (교체) ───────────────────
"""
raw B_{n}.csv  (n×n, 헤더 없음)  →  B_{n}_labeled.csv
행·열 인덱스에 '가동 중 발전기 버스 번호'를 달아 저장.

전제
----
make_B.py 실행 시
   np.save("data/bus_id_red.npy", bus_id_red)   # 189 버스
   np.save("data/gen_mask_active.npy", gen_mask_active)  # 가동 발전기 True
를 저장해 두었다고 가정.

사용
----
python scripts/add_headers_to_B.py
(또는 --in / --out 커스텀 지정)
"""
from pathlib import Path
import argparse, numpy as np, pandas as pd

# ─── main ───────────────────────────────────────────────────
def main(in_csv: Path, out_csv: Path,
         bus_id_file: Path, mask_file: Path):

    B = np.loadtxt(in_csv, delimiter=",")                # (n,n)
    bus_id_red = np.load(bus_id_file)                    # (189,)
    gen_mask   = np.load(mask_file)                      # True for ACTIVE gens
    gen_buses  = bus_id_red[gen_mask]                    # (n,)

    if B.shape[0] != len(gen_buses):
        raise SystemExit(f"❌  size mismatch: B is {B.shape[0]}×, "
                         f"gen_buses={len(gen_buses)}. "
                         "make_B.py & mask npy 파일이 일치하는지 확인.")

    df = pd.DataFrame(B, index=gen_buses, columns=gen_buses)
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, float_format="%.12g")
    print(f"✅  saved → {out_csv}  (shape={B.shape})")

# ─── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="in_csv",
                   default="data/B_41.csv",
                   help="raw B matrix csv")
    p.add_argument("--out", dest="out_csv",
                   default="data/B_41_labeled.csv",
                   help="labeled csv output")
    p.add_argument("--bus", dest="bus_id",
                   default="data/bus_id_red.npy",
                   help="bus_id_red.npy path")
    p.add_argument("--mask", dest="mask",
                   default="data/gen_mask_active.npy",
                   help="gen_mask_active.npy path (STATUS==1)")
    args = p.parse_args()
    main(Path(args.in_csv), Path(args.out_csv),
         Path(args.bus_id), Path(args.mask))
# ───────────────────────────────────────────────────────────
