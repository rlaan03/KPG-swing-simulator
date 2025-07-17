# webapp/app.py
# KPG-193 Swing-Equation Step-Disturbance Lab   (UI/UX 개선판)

import sys, pathlib, streamlit as st, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from simcore import run_step_disturbance            # 계산 API
import networkx as nx
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events   # pip install streamlit-plotly-events
import numpy as np
import numpy.linalg as la
import plotly.express as px
import re



# ────────────────── 페이지 설정 ──────────────────
st.set_page_config(
    page_title="KPG-193 Swing Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"   # 사이드바 자동 숨김
)
st.title("KPG-193 Swing Equation – Step-Disturbance Lab")


# ────────────────── 세션 상태 초기화 ──────────────────
if "sim_done" not in st.session_state:      # ← 새 줄
    st.session_state.sim_done = False

if "events" not in st.session_state:      # [(bus, t_step, ΔPm), …]
    st.session_state.events = []

if "selected_buses" not in st.session_state:
    st.session_state.selected_buses = []   # 4-slot용


# ────────────────── 유틸 함수들 ──────────────────
@st.cache_resource
def load_power_data():
    # 1) gen_result_with_type.csv 로드 (data/ 또는 pf/ 아래에서 찾아서)
    df_gen = None
    for gen_path in (ROOT/"data"/"gen_result_with_type.csv",
                     ROOT/"pf"/"gen_result_with_type.csv"):
        if gen_path.exists():
            df_gen = pd.read_csv(gen_path)
            df_gen.columns = df_gen.columns.str.lower()
            break
    if df_gen is None:
        st.error("❌ gen_result_with_type.csv를 찾을 수 없습니다.")
        st.stop()

    # 2) MATPOWER .m 파일에서 mpc.bus 블록 파싱
    mfile = (ROOT / "KPG193_ver1_2" / "KPG193_ver1_2"
                     / "network" / "m" / "KPG193_ver1_2.m")
    bus_pd = {}
    inside = False
    with open(mfile, encoding="utf-8") as f:
        for line in f:
            if not inside:
                if re.match(r"\s*mpc\.bus\s*=\s*\[", line):
                    inside = True
                continue
            if re.match(r"\s*\];", line):
                break
            if ";" not in line or not line.strip():
                continue
            # 숫자 파트만 분리
            num_part = line.split(";", 1)[0]
            tokens   = num_part.strip().split()
            if len(tokens) >= 3:
                bus = int(float(tokens[0]))
                pd_val = float(tokens[2])    # 3번째 항목이 Pd
                bus_pd[bus] = pd_val

    # DataFrame으로
    df_bus = pd.DataFrame({
        "bus": list(bus_pd.keys()),
        "pd":  list(bus_pd.values())
    })

    return df_gen, df_bus

df_gen, df_bus = load_power_data()


@st.cache_resource
def load_B_and_Z():
    # ── 1) m-file에서 mpc.branch 블록 파싱 ──
    mfile = (ROOT / "KPG193_ver1_2" / "KPG193_ver1_2"
                   / "network" / "m" / "KPG193_ver1_2.m")
    branch_data = []
    inside = False
    with open(mfile, encoding="utf-8") as f:
        for line in f:
            if not inside:
                if re.match(r"\s*mpc\.branch\s*=\s*\[", line):
                    inside = True
                continue
            if re.match(r"\s*\];", line):
                break
            # 주석 제거, 빈 줄 무시
            if ";" not in line or line.strip().startswith("%"):
                continue
            # 숫자 토큰만 분리
            nums = re.split(r"[;\s]+", line.strip())
            # 최소 4개: fbus, tbus, r, x
            if len(nums) >= 4:
                fbus = int(float(nums[0]))
                tbus = int(float(nums[1]))
                r    = float(nums[2])
                x    = float(nums[3])
                branch_data.append((fbus, tbus, r, x))

    # ── 2) B행렬(순수 susceptance) 구성 ──
    buses = sorted({b for row in branch_data for b in row[:2]})
    idx   = {bus:i for i,bus in enumerate(buses)}
    n     = len(buses)
    B     = np.zeros((n,n), float)
    for fbus, tbus, r, x in branch_data:
        i = idx[fbus]; j = idx[tbus]
        b_ij = -1.0 / x
        B[i,j] += b_ij
        B[j,i] += b_ij
        B[i,i] -= b_ij
        B[j,j] -= b_ij

    # ── 3) 의사역행렬 Z 계산 ──
    Z = la.pinv(B)
    return buses, B, Z

BUSES, B_mat, Z_bus = load_B_and_Z()



def electrical_distance(from_buses):
    """
    from_buses: 외란 노드 리스트
    Buses, Z_bus: 전역 변수로 위에서 로드된 값 사용
    """
    buses = BUSES
    Z = Z_bus
    # 각 from_bus에 대해 거리 벡터 계산
    dist_list = []
    for fb in from_buses:
        k = buses.index(fb)
        # d_i = Z_kk + Z_ii - 2 Z_ik
        diagZ = np.diag(Z)
        d = Z[k,k] + diagZ - 2 * Z[:,k]
        dist_list.append(d)
    # 여러 외란노드인 경우 평균
    avg_dist = np.mean(dist_list, axis=0)
    return pd.DataFrame({"bus": buses, "distance": avg_dist})



# ────────────────── 네트워크 로드 ──────────────────
@st.cache_resource
def load_network():
    base = ROOT / "KPG193_ver1_2" / "KPG193_ver1_2" / "network"
    loc  = pd.read_csv(base / "location" / "bus_location.csv")
    br   = pd.read_csv(base / "branch"   / "branch.csv")

    loc.columns = loc.columns.str.lower()
    def pick(keys): return next(c for c in loc.columns
                                if any(k in c for k in keys))
    id_col = pick(["bus"])
    x_col  = pick(["x", "lon"])
    y_col  = pick(["y", "lat"])

    G   = nx.from_pandas_edgelist(br, "fbus", "tbus")
    pos = {getattr(r, id_col): (getattr(r, x_col), getattr(r, y_col))
           for r in loc.itertuples(index=False)}
    return G, pos

G, pos = load_network()

def bus_to_name_safe(b):
    try:
        return bus2name.get(int(float(b)), "")
    except Exception:
        return ""

def load_bus_names():
    base = ROOT / "KPG193_ver1_2" / "KPG193_ver1_2" / "network" / "location"
    # 실제 존재하는 bus_location.csv 를 읽어 오도록 수정
    loc = pd.read_csv(base / "bus_location.csv")
    loc.columns = loc.columns.str.lower()

    # 나머지 로직 동일
    id_col = next(c for c in loc.columns if "bus" in c)
    name_col = next(c for c in loc.columns 
                    if any(k in c for k in ["지점", "name", "desc", "location"]))

    bus2name = {}
    for r in loc.itertuples(index=False):
        bus_val  = getattr(r, id_col)
        name_val = getattr(r, name_col)
        if pd.notna(bus_val) and pd.notna(name_val) and str(bus_val).strip():
            try:
                bus2name[int(bus_val)] = str(name_val)
            except:
                pass
    return bus2name


# *** 이 부분 반드시 표 코드보다 위에 위치 ***
bus2name = load_bus_names()

def show_results(res):
    """ 시뮬레이션 결과( dict )를 화면에 그린다 """
    df_f = (pd.DataFrame(res["freq"].T,
                         columns=[f"Bus {b}" for b in res["buses"]])
              .assign(t=res["t"])
              .set_index("t"))
    st.line_chart(df_f, height=400)

def show_quick_guide():
    st.info("🔰 먼저 외란 이벤트를 추가한 뒤 ‘Run Simulation’ 버튼을 눌러주세요.")

def show_system_metrics(res):
    import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go

    # ─ 1) Node-level 지표 계산 ─
    freq      = res["freq"]       # shape (n_buses, n_t)
    roc       = res["roc"]
    disturbed = {e[0] for e in st.session_state.events}
    buses     = np.array(res["buses"], dtype=int)
    mask      = np.array([b not in disturbed for b in buses])

    nadirs_mhz = freq.min(axis=1)[mask] * 1e3
    rocofs_mhz = np.abs(roc).max(axis=1)[mask] * 1e3
    df_metrics = pd.DataFrame({
        "Bus": buses[mask],
        "Nadir (mHz)": nadirs_mhz,
        "RoCoF_max (mHz/s)": rocofs_mhz,
    })

    # ─ 2) Kuramoto ΔR(t) 계산 ─
    t_vec      = res["t"]
    R_ts       = np.array(res["R"])
    R0         = R_ts[0]
    delta_R_mr = (R_ts - R0) * 1e3
    max_dev    = np.max(np.abs(delta_R_mr)) * 1.1

    # ─ 3) System-level Metrics 카드 ─
    avg_fd = np.mean(np.abs(freq))        # 전체 |Δf| 평균
    min_R  = float(np.min(R_ts))          # 최소 R(t)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Average |Δf|", f"{avg_fd:.3e} Hz")
    with c2:
        st.metric("Minimum R(t)", f"{min_R:.4f}")
    st.markdown("---")

    # ─ 4) Node-level Distributions ─
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Frequency Nadir (mHz)")
        fig_n = px.histogram(df_metrics, x="Nadir (mHz)", nbins=20)
        mean_n = df_metrics["Nadir (mHz)"].mean()
        fig_n.add_vline(
            x=mean_n, line_dash="dash",
            annotation_text=f"Mean={mean_n:.2f} mHz",
            annotation_position="top right"
        )
        st.plotly_chart(fig_n, use_container_width=True)

    with col2:
        st.subheader("RoCoFₘₐₓ (mHz/s)")
        fig_r = px.histogram(df_metrics, x="RoCoF_max (mHz/s)", nbins=20)
        mean_r = df_metrics["RoCoF_max (mHz/s)"].mean()
        fig_r.add_vline(
            x=mean_r, line_dash="dash",
            annotation_text=f"Mean={mean_r:.2f} mHz/s",
            annotation_position="top right"
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col3:
        st.subheader("Kuramoto R(t) (mR)")
        fig_dR = go.Figure()
        fig_dR.add_trace(go.Scatter(x=t_vec, y=delta_R_mr, mode="lines"))
        fig_dR.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="R (mR)",
            yaxis=dict(range=[-max_dev, max_dev]),
            height=300, margin=dict(l=40, r=20, t=30, b=30)
        )
        st.plotly_chart(fig_dR, use_container_width=True)


def show_distance_analysis(res):
    import pandas as pd, plotly.express as px

    # ─ 1) 전기적 거리 계산 ─
    disturbed = [e[0] for e in st.session_state.events]
    df_dist   = electrical_distance(disturbed)

    # ─ 2) 같은 df_metrics 재생성 (Nadir만 필요) ─
    import numpy as np
    freq      = res["freq"]
    buses     = np.array(res["buses"], dtype=int)
    mask      = np.array([b not in disturbed for b in buses])
    nadirs_mhz= freq.min(axis=1)[mask] * 1e3
    df_metrics = pd.DataFrame({
        "Bus": buses[mask],
        "Nadir (mHz)": nadirs_mhz
    })

    # ─ 3) 조인하여 히스토그램 준비 ─
    df_joint = pd.merge(df_dist, df_metrics,
                        left_on="bus", right_on="Bus")

    # ─ 4) 그래프 배치 ─
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📐 Electrical Distance Distribution")
        fig_d = px.histogram(
            df_dist, x="distance", nbins=30,
            title="Electrical Distance (p.u. impedance)",
            color_discrete_sequence=["#61AFEF"]
        )
        fig_d.update_layout(
            xaxis_title="Electrical Distance (p.u. impedance)",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig_d, use_container_width=True)

    with c2:
        st.subheader("📈 Disturbance vs Distance")
        fig_hd = px.density_heatmap(
            df_joint,
            x="distance", y="Nadir (mHz)",
            nbinsx=30, nbinsy=20,
            color_continuous_scale="Cividis"
        )
        fig_hd.update_layout(
            xaxis_title="Electrical Distance (p.u. impedance)",
            yaxis_title="Frequency Nadir (mHz)",
            coloraxis_colorbar=dict(title="Count")
        )
        st.plotly_chart(fig_hd, use_container_width=True)



def render_selected_plots():
    st.subheader("🎯 선택 노드 주파수 응답 (Δf) — (아래로 스크롤)")
    if st.session_state.selected_buses:
        if st.button("🧹 슬롯 전체 비우기", key="reset_slots"):
            st.session_state.selected_buses.clear()
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            else:
                st.rerun()
            return  # 반드시 return! 아래 코드 안 돌게

    st.markdown("""
        <style>
        .scroll-area {
            height: 850px;
            overflow-y: auto;
            border-radius: 12px;
            border: 1.5px solid #333;
            background: #181c24;
            padding: 14px;
            margin-bottom: 8px;
        }
        </style>
        <div class="scroll-area">
        """, unsafe_allow_html=True)

    if st.session_state.selected_buses and st.session_state.get("sim_done") and "sim_res" in st.session_state:
        res = st.session_state.sim_res
        for i, b in enumerate(st.session_state.selected_buses):
            if b in res["buses"]:
                j = list(res["buses"]).index(b)
                df_b = pd.DataFrame({"Δf": res["freq"][j], "t": res["t"]}).set_index("t")
                with st.container():
                    st.markdown(f"<b>Bus {b}</b>", unsafe_allow_html=True)
                    st.line_chart(df_b, height=180)
    st.markdown("</div>", unsafe_allow_html=True)










# ────────────────── 컨트롤 바 ──────────────────
with st.container():
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        tot_time = st.number_input("Total time (s)", 5.0, 120.0, 20.0, 1.0,
                                   key="tot_time")
    with col2:
        dt = st.number_input("dt (s)", 0.005, 0.1, 0.05, 0.005,
                             key="dt")
    with col3:
        run_btn = st.button("🚀 Run Simulation",
                            disabled=not st.session_state.get("events"))
    with col4:
        st.markdown(" ")   # (테마·모드 토글 자리)



@st.cache_resource
def draw_network(events, flash_id=None, size_base=9):
    """발전기 노드만 클릭 허용.
       flash_id: 직전 클릭 버스 → marker를 잠깐 키워 피드백"""
    # ─── 41-bus 발전기 집합 한 번만 구해 둠 ─────────────────────────
    if "gen_buses" not in st.session_state:
        from simcore.step_disturbance import load_system
        _, _, _, _, _, gen41 = load_system()  # 41개 버스 번호
        st.session_state.gen_buses = set(gen41.tolist())
    gen_set = st.session_state.gen_buses
    affected = {e[0] for e in events}

    node_x, node_y, color, size, ids, hover_texts = [], [], [], [], [], []
    for n, (x,y) in pos.items():
        node_x.append(x); node_y.append(y); ids.append(n)

        # ─── 여기에 원하는 정보 조합 ──────────────────
        region = bus2name.get(n, "")
        # Pg 합계
        dfg   = df_gen[df_gen["bus"] == n]
        pg    = dfg["pg"].sum() if not dfg.empty else 0.0
        # 발전기 타입들
        types = ", ".join(dfg["type"].unique()) if not dfg.empty else "Load"
        # Pd
        dfb   = df_bus[df_bus["bus"] == n]
        pd_val= float(dfb["pd"].iloc[0]) if not dfb.empty else 0.0
        pd_val = pd_val / 100.0

        hover_texts.append(
            f"Bus {n}<br>"
            f"Region: {region}<br>"
            f"Pg: {pg:.3f} pu<br>"
            f"Pd: {pd_val:.3f} pu<br>"
            f"Gen Type: {types}"
        )
        # ─────────────────────────────────────────────

        # 색상·크기 규칙
        if n in gen_set:
            base_col = "skyblue"
            if n in affected:
                base_col = "red"
            color.append(base_col)
            size.append(size_base * (1.4 if n == flash_id else 1))
        else:
            color.append("#C0C0C0")
            size.append(size_base * 0.7)

    # ─── 간선 좌표 ──────────────────────────────────────────────
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # ─── Plotly 그리기 ─────────────────────────────────────────
    fig = go.Figure()
    # 1) 간선
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.4, color="gray"),
        hoverinfo="none"
    ))
    # 2) 노드
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=size, color=color,
                    line=dict(width=0.6, color="gray")),
        customdata=ids,
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_white",
        height=850,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(range=[125, 131])

    return fig



# ────────────────── 메인 1행 ──────────────────
net_col, info_col = st.columns([0.7, 0.3])

with net_col:
    st.subheader("⚡ 한국 전력망 네트워크")
    fig = draw_network(st.session_state.events,
                       flash_id=st.session_state.get("flash"))
    clicks = plotly_events(fig, click_event=True, hover_event=False,
                           override_height=720, key="graph")

with info_col:
    if not st.session_state.get("sim_done"):
        show_quick_guide()

        # ───────── 외란 이벤트 입력 ─────────
        if clicks:
            idx    = clicks[0]["pointIndex"]
            bus_id = list(pos.keys())[idx]
            if bus_id not in st.session_state.gen_buses:
                st.info(f"Bus {bus_id} 는 발전기가 없는 노드입니다.")
            else:
                with st.expander(f"Add disturbance to Bus {bus_id}", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        t_val = st.number_input(
                            "t_step (s)",
                            min_value=0.0, max_value=tot_time,
                            value=5.0, step=0.1,
                            key=f"t_inp_{bus_id}"
                        )
                    with c2:
                        dp_val = st.number_input(
                            "ΔPm (pu)",
                            min_value=-15.0, max_value=15.0,
                            value=-1.0, step=0.05,
                            key=f"dp_inp_{bus_id}"
                        )
                    if st.button("➕ Add", key=f"add_{bus_id}"):
                        st.session_state.events.append((bus_id, t_val, dp_val))
                        st.success(f"Bus {bus_id} 외란 추가 (t={t_val}s, ΔPm={dp_val} pu)")

        # ───────── 이벤트 목록 / Clear 버튼 ─────────
        if st.session_state.events:
            df_ev = pd.DataFrame(st.session_state.events,
                                 columns=["Bus","t_step","ΔPm"])
            st.table(df_ev)
            if st.button("🗑️ Clear all disturbances"):
                st.session_state.events.clear()

    else:
        render_selected_plots()




# ───────── 캐싱된 시뮬레이션 함수 ─────────
@st.cache_data(show_spinner=False)
def cached_sim(events_tuple, t_tot, step):
    return run_step_disturbance(list(events_tuple), t_tot, step)

# ───────── 실행 & 결과 ─────────

if run_btn:
    wait_box = st.empty()
    wait_box.info("🔄  Running simulation… please wait")
    with st.spinner("Running swing simulation …"):
        res = cached_sim(tuple(st.session_state.events), tot_time, dt)
    wait_box.empty()
    st.session_state.sim_res  = res
    st.session_state.sim_done = True




# ────────────────── 결과 탭 ──────────────────
if st.session_state.get("sim_done") and "sim_res" in st.session_state:
    res = st.session_state.sim_res
    st.markdown("---")

    tab_f, tab_m, tab_d = st.tabs(["Δf Overview", "Metrics", "Distance"])

    with tab_f:
        show_results(res)

    with tab_m:
        show_system_metrics(res)

    with tab_d:
        show_distance_analysis(res)

