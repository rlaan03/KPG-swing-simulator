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


# webapp/app.py 상단, import 다음쯤에
from simcore.step_disturbance import load_system

if "gen_buses" not in st.session_state:
    # load_system() 을 호출해서 41개 발전기 bus 리스트를 뽑아옵니다
    _, _, _, _, _, gen41 = load_system()
    st.session_state.gen_buses = set(gen41.tolist())


if "sim_done" not in st.session_state:      # ← 새 줄
    st.session_state.sim_done = False

    
import re

# ────────────────── 발전기/부하 데이터 로드 ──────────────────
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
            height: 750px;
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


# ────────────────── 세션 상태 ──────────────────
if "events" not in st.session_state:      # [(bus, t_step, ΔPm), …]
    st.session_state.events = []

if "selected_buses" not in st.session_state:
    st.session_state.selected_buses = []   # 4-slot용


# ────────────────── 페이지 설정 ──────────────────
st.set_page_config(
    page_title="KPG-193 Swing Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* 사이드바가 펼쳐졌을 때 폭을 300px로 고정 */
      [data-testid="stSidebar"][aria-expanded="true"] {
        width: 300px;
      }
      /* 사이드바가 접혔을 때 폭을 80px로 고정 */
      [data-testid="stSidebar"][aria-expanded="false"] {
        width: 80px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("KPG-193 Swing Equation – Step-Disturbance Lab")

st.markdown("""
    <style>
    /* 전체 padding 제거 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* 헤더 위쪽 여백 제거 */
    header[data-testid="stHeader"] {
        margin-top: -2rem;
    }

    /* 좌측 사이드바 폭 줄이기 */
    section[data-testid="stSidebar"] {
        width: 280px;
    }
    </style>
""", unsafe_allow_html=True)


# ── Sidebar 맨 위에 추가 ───────────────────────────────────
if st.sidebar.button("🔄 Reset All"):
    for k in ("events", "flash"):
        st.session_state.pop(k, None)   # 키가 없어도 무시
    st.rerun()


# ───────── 사이드바: 시뮬 파라미터 ─────────
st.sidebar.header("Simulation Params")
tot_time = st.sidebar.number_input("Total time (s)", 5.0, 120.0, 20.0, 1.0)
dt       = st.sidebar.number_input("dt (s)", 0.005, 0.1, 0.05, 0.005)

st.sidebar.divider()
if st.sidebar.button("Clear disturbances"):
    st.session_state.events.clear()            # 외란 목록 비우기
    st.session_state.sim_done = False          # 분석-모드 → 입력-모드
    st.session_state.pop("sim_res", None)      # 이전 결과 제거
    st.session_state.pop("flash",   None)      # 확대 표시 초기화
    st.rerun()                                 # 화면 새로고침                      # 즉시 화면 리프레시

st.sidebar.subheader(f"Disturbances ({len(st.session_state.events)})")
if st.session_state.events:
    df_ev = (pd.DataFrame(st.session_state.events,
                          columns=["Bus","t_step","ΔPm"])
                .reset_index(drop=True))
    
    # 열 이름 한글로 변경
    df_ev = df_ev.rename(columns={
        "t_step": "외란 시점",
        "ΔPm":    "출력 변화량"
    })
    # Bus 컬럼 타입이 float이면 int로 변환
    df_ev["Bus"] = df_ev["Bus"].apply(lambda x: int(x) if pd.notna(x) else x)
    # '지명' 컬럼 추가 (int 변환 실패 시 빈 문자열)
    df_ev["지명"] = df_ev["Bus"].map(lambda b: bus2name.get(b, "") if pd.notna(b) and b in bus2name else "")
    # 순서: #, Bus, 지명, t_step, ΔPm
    df_ev = df_ev[["#", "Bus", "지명", "t_step", "ΔPm"]]

    st.sidebar.dataframe(
        df_ev,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Bus":        st.column_config.NumberColumn("Bus", width="small"),
            "지점/지명":   st.column_config.TextColumn("지점/지명", width="medium"),
            "외란 시점":   st.column_config.NumberColumn("외란 시점", width="small"),
            "출력 변화량": st.column_config.NumberColumn("출력 변화량", width="small"),
        },
        height=110 + 32 * len(df_ev)
    )



ready_btn = st.sidebar.button("Ready ↘", disabled=not st.session_state.events)
run_btn   = st.sidebar.button("Run Simulation 🚀",
                              disabled=not ready_btn)

# ────────────── Bus → Name(지명) 매핑 생성 ──────────────
@st.cache_resource
@st.cache_resource



@st.cache_resource
def draw_network(events, flash_id=None, size_base=9):
    """발전기 노드만 클릭 허용.
       flash_id: 직전 클릭 버스 → marker를 잠깐 키워 피드백"""
    # ─── 41-bus 발전기 집합 한 번만 구해 둠 ─────────────────────────
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
        height=750,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(
        range=[126, 130],
        showticklabels=False,    # ← X축 숫자 숨김
    )
    fig.update_yaxes(
        range=[34, 38.4],
        showticklabels=False,    # ← Y축 숫자 숨김
    )
    return fig






# ── 네트워크 & 상세 그래프를 좌·우 2-칼럼 배치 ───────────────────
cols = st.columns([2,3])
with cols[0]:
    st.subheader("⚡ 한국 전력망 네트워크")
    fig = draw_network(st.session_state.events, flash_id=st.session_state.get("flash"))
    # 여기서 clicks 정의!
    clicks = plotly_events(fig, click_event=True, hover_event=False, override_height=750, key="graph")



import streamlit.components.v1 as components
import plotly.graph_objects as go

import streamlit.components.v1 as components
import plotly.graph_objects as go


with cols[1]:
    # ── 동적 헤더 결정 ──
    if not st.session_state.sim_done:
        header_text = "📘Swing Simulator Quickstart"
    elif not st.session_state.selected_buses:
        header_text = "🖱️ 발전기 노드 선택 대기 중"
    else:
        header_text = "🎯 선택 노드 주파수 응답 (Δf)"
    st.subheader(header_text)

    # 1) 아직 시뮬레이션을 안 돌렸으면 Read-Me 박스 출력
    if not st.session_state.sim_done:
        st.markdown(
            """
            <style>
            .guide {
                background: #2d3035;
                border-radius: 8px;
                padding: 24px;
                color: #eee;
            }
            .guide h2 {
                margin: 0 0 16px;
                font-size: 1.4em;
                display: flex;
                align-items: center;
            }
            .guide h2 .emoji {
                margin-right: 8px;
            }
            .step-card {
                display: flex;
                align-items: flex-start;
                background: #3a3d42;
                border-radius: 6px;
                padding: 12px 16px;
                margin-bottom: 12px;
            }
            .step-number {
                flex: 0 0 32px;
                height: 32px;
                line-height: 32px;
                text-align: center;
                font-weight: bold;
                background: #61afef;
                border-radius: 50%;
                margin-right: 12px;
            }
            .step-content strong {
                color: #61afef;
            }
            .step-content p {
                margin: 4px 0 0;
                line-height: 1.5;
            }
            .step-content code {
                display: inline-block;
                margin: 0 6px;
                padding: 2px 4px;
                background: rgba(255,255,255,0.1);
                border-radius: 3px;
                font-size: 0.95em;
            }
            </style>
            
            <div class="guide">
            <div class="step-card">
                <div class="step-number">1</div>
                <div class="step-content">
                <strong>네트워크 노드 확인</strong>
                <p>파란색 노드는 한 개 이상의 발전기를 포함하는 노드, 회색 노드는 발전기가 없는 부하 노드입니다.</p>
                </div>
            </div>
        
            <div class="step-card">
                <div class="step-number">2</div>
                <div class="step-content">
                <strong>외란 노드 선택</strong>
                <p>파란색 노드 중에서 외란을 줄 노드를 클릭하세요. 웹페이지 하단에 해당 노드에 대한 외란 설정 창이 표시됩니다.</p>
                </div>
            </div>

            <div class="step-card">
                <div class="step-number">3</div>
                <div class="step-content">
                <strong>외란 설정</strong>
                <p>
                    해당 노드에 외란을 부여할 시점(t_step) 과 출력 변화(ΔPm) 설정 후 Add 버튼을 클릭하면, 사이드바에 이벤트가 저장됩니다.<br><br>
                    All Off 버튼은 해당 노드의 모든 발전기의 출력을 0으로 합니다.
                    첫 외란 저장 이후에도 그림에서 노드를 클릭하여 추가적으로 외란을 가할 수 있습니다.<br><br>
                    ΔPm의 1.0pu는 실계통에서 100MW의 출력변화를 의미합니다.
                </p>
                </div>
            </div>

            <div class="step-card">
                <div class="step-number">4</div>
                <div class="step-content">
                <strong>시뮬레이션 준비</strong>
                <p>
                    사이드바 상단에서 시뮬레이션 시간(Total time), 시뮬레이션 스텝(dt)을 설정하세요.<br><br>
                    최종 검토 이후 Ready 버튼을 클릭하면 Run Simulation 🚀 활성화되고, 시뮬레이션을 실행할 수 있습니다.
                </p>
                </div>
            </div>

            <div class="step-card">
                <div class="step-number">5</div>
                <div class="step-content">
                <strong>결과 확인</strong>
                <p>페이지 하단에 전체 주파수 응답, R(t), RoCoF 지표가 출력됩니다.</p>
                </div>
            </div>

            <div class="step-card">
                <div class="step-number">6</div>
                <div class="step-content">
                <strong>노드별 주파수 거동 보기</strong>
                <p>
                    시뮬레이션 종료 이후, 그림에서 원하는 노드(발전기를 포함하는 파란색 노드)를 클릭하면,
                    해당 노드와 외란을 부여한 노드들의 주파수 변동(Δf) 시계열 그래프가 추가됩니다. <br><br>
                    ※현재의 안내페이지는 사라집니다.
                </p>
                </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )




    else:
        # 2) 시뮬레이션은 끝났는데 선택된 노드가 없으면 안내
        if not st.session_state.selected_buses:
            st.markdown(
                "<div style='padding:20px; color:#aaa;'>"
                "발전기 노드를 클릭하여 주파수 응답을 확인하세요."
                "</div>",
                unsafe_allow_html=True
            )
        # 3) 시뮬 & 노드 선택 모두 된 경우에만 스크롤 영역에 그래프 삽입
        else:
            slot_html = """
            <style>
              .scroll-area {
                height: 750px;
                overflow-y: auto;
                border-radius: 12px;
                border: 1.5px solid #333;
                background: #181c24;
                padding: 14px;
                margin-bottom: 8px;
              }
            </style>
            <div class="scroll-area">
            """
            res = st.session_state.sim_res
            for b in st.session_state.selected_buses:
                if b in res["buses"]:
                    j    = list(res["buses"]).index(b)
                    df_b = pd.DataFrame({"Δf": res["freq"][j],
                                         "t":  res["t"]}).set_index("t")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_b.index, y=df_b["Δf"],
                                             mode="lines"))
                    fig.update_layout(height=180,
                                      margin=dict(l=20,r=20,t=20,b=20),
                                      xaxis=dict(showgrid=False),
                                      yaxis=dict(showgrid=False))
                    plot_html = fig.to_html(full_html=False,
                                            config={"displayModeBar":False})
                    slot_html += f"""
                      <div style='margin-bottom:16px;'>
                      <div style="color: #FFFFFF; font-weight: bold; margin-bottom: 6px;">
                        <b>Bus {b} {bus_to_name_safe(b)}</b>
                        <div>
                        {plot_html}
                      </div>
                    """
            slot_html += "</div>"
            components.html(slot_html, height=750)








if clicks:
    idx = clicks[0]["pointIndex"]
    bus_id = list(pos.keys())[idx]
    # slot에 bus 추가 (중복 방지, 최대 n개)
    if bus_id not in st.session_state.selected_buses:
        st.session_state.selected_buses.append(bus_id)
        st.rerun()


    # 시뮬 후(분석 모드)에는 노드 클릭 무시



    if clicks:
        idx = clicks[0]["pointIndex"]          # 0‥192 중 하나
        bus_id = list(pos.keys())[idx]

        # 발전기 노드가 아니면 무시
        if bus_id not in st.session_state.gen_buses:
            st.info(f"Bus {bus_id} 는 발전기가 없는 노드입니다.")
        else:
            st.session_state.flash = bus_id    # - 확대 효과
            with st.expander(f"Add disturbance to Bus {bus_id}", expanded=True):
                # 1) 입력 칸 2단
                c1, c2 = st.columns(2)
                with c1:
                    t_val = st.number_input(
                        "t_step (s)",
                        min_value=0.0,
                        max_value=tot_time,
                        value=5.0,
                        step=0.1,
                        key=f"t_inp_{bus_id}"
                    )
                with c2:
                    # 2) ΔPm 입력 영역을 3:1 비율로 나눠서 우측에 All Off 버튼
                    pm_col, off_col = st.columns([3, 1])
                    with pm_col:
                        dp_val = st.number_input(
                            "ΔPm (pu)",
                            min_value=-15.0,
                            max_value=15.0,
                            value=-1.0,
                            step=0.05,
                            key=f"dp_inp_{bus_id}"
                        )
                    with off_col:
                        # All Off 버튼 눌리면 해당 버스의 전체 Pg 합계를 읽어 -PgSum 만큼 이벤트 추가
                        if st.button(
                            "🔌 All Off",
                            help="해당 버스의 모든 발전기를 꺼서 Pm을 총 pg만큼 음수로 설정합니다.",
                            key=f"off_{bus_id}"
                        ):
                            pg_sum = float(df_gen.loc[df_gen["bus"] == bus_id, "pg"].sum())
                            st.session_state.events.append((bus_id, float(t_val), -pg_sum))
                            del st.session_state.flash
                            st.success(
                                f"Bus {bus_id} 전체 발전기 OFF (ΔPm = {-pg_sum:.3f} pu)"
                            )
                            # Streamlit rerun
                            if hasattr(st, "rerun"):
                                st.rerun()
                            else:
                                st.experimental_rerun()

                # 3) Add 버튼은 그대로 아래
                if st.button("➕ Add", key=f"add_{bus_id}_{t_val}_{dp_val}"):
                    st.session_state.events.append((bus_id, float(t_val), float(dp_val)))
                    del st.session_state.flash
                    st.success(f"Bus {bus_id} 외란 추가 (t={t_val}s, ΔPm={dp_val} pu)")
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()




# ───────── 캐싱된 시뮬레이션 함수 ─────────
@st.cache_data(show_spinner=False)
def cached_sim(events_tuple, t_tot, step):
    return run_step_disturbance(list(events_tuple), t_tot, step)

# ───────── 실행 & 결과 ─────────
# ───────── 실행 & 결과 저장 ─────────
# ───────── 실행 & 결과 저장 ─────────
if run_btn:
    wait_box = st.empty()
    wait_box.info("🔄  Running simulation… please wait")
    with st.spinner("Running swing simulation …"):
        res = cached_sim(tuple(st.session_state.events), tot_time, dt)
    wait_box.empty()
    st.session_state.sim_res  = res
    st.session_state.sim_done = True

# ───────── 결과 표시 ─────────
if st.session_state.sim_done and "sim_res" in st.session_state:
    res = st.session_state.sim_res

    # 1) 전체 네트워크 응답
    st.header("🌐 전체 네트워크 응답")
    st.caption("모든 버스의 주파수 시계열을 한번에 확인합니다.")
    show_results(res)

    # 2) Node-level 지표 계산
    import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go

    freq      = res["freq"]       # (n_buses, n_t)
    roc       = res["roc"]        # same shape
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

    # 3) Kuramoto ΔR(t) 계산
    t_vec      = res["t"]
    R_ts       = np.array(res["R"])
    R0         = R_ts[0]
    delta_R_mr = (R_ts - R0) * 1e3
    max_dev    = np.max(np.abs(delta_R_mr)) * 1.1

    # 4) System-level Metrics
    avg_fd = np.mean(np.abs(freq))        # 전체 |Δf| 평균 (Hz)
    min_R  = float(np.min(R_ts))          # 최소 R(t)

    st.header("🔧 System-level Metrics")
    st.caption("전력망의 외란에 대한 대표 지표를 요약합니다.")
    col_sys1, col_sys2 = st.columns(2)
    with col_sys1:
        st.metric("Average |Δf|", f"{avg_fd:.3e} Hz")
        st.caption("전체 버스들의 주파수 편차-시간 평균")
    with col_sys2:
        st.metric("Minimum R(t)", f"{min_R:.4f}")
        st.caption("동기화 Kuramoto 지표 R(t)의 최저값")

    st.markdown("---")  # 구분선

    # 5) Node-level Distributions
    st.header("📊 Node-level Distributions")
    st.caption("외란을 받지 않은 발전기 노드들의 주요 지표 분포입니다.")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Frequency Nadir (mHz)")
        st.caption("각 노드 주파수가 가장 낮게 떨어진 최저값 분포")
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
        st.caption("각 노드 최대 주파수 변화율(RoCoF) 분포")
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
        st.caption("동기화 지표 R(t)의 절대 변화량 시계열 (mR)")
        fig_dR = go.Figure()
        fig_dR.add_trace(go.Scatter(x=t_vec, y=delta_R_mr, mode="lines"))
        fig_dR.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="R (mR)",
            yaxis=dict(range=[-max_dev, max_dev]),
            height=300, margin=dict(l=40, r=20, t=30, b=30)
        )
        st.plotly_chart(fig_dR, use_container_width=True)


    # ────────── 6) Disturbance ↔ Distance & 2D Histogram ──────────
    # Sidebar에서 컬러맵 선택
    custom_color = "#61AFEF"

    heatmap_scale = st.sidebar.selectbox(
        "Heatmap 컬러스케일", 
        ["Viridis", "Cividis", "Plasma", "Inferno", "Magma"], 
        index=1
    )

    # 데이터 준비
    disturbed_buses = [e[0] for e in st.session_state.events]
    df_dist = electrical_distance(disturbed_buses)
    # df_metrics는 위에서 생성된 DataFrame(["Bus","Nadir (mHz)",...])
    df_joint = pd.merge(
        df_dist,
        df_metrics[["Bus", "Nadir (mHz)"]],
        left_on="bus", right_on="Bus"
    )

    # 한 줄에 반반씩 배치
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📐 Electrical Distance Distribution")
        st.caption("외란 노드로부터 다른 노드들의 전기적 거리 분포")
        fig_d = px.histogram(
            df_dist,
            x="distance",
            nbins=30,
            title="Electrical Distance (p.u. impedance)",
            color_discrete_sequence=[custom_color]
        )
        fig_d.update_layout(
            xaxis_title="Electrical Distance (p.u. impedance)",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig_d, use_container_width=True)

    with col2:
        st.subheader("📈 Disturbance Magnitude vs Electrical Distance")
        st.caption("주파수 Nadir(mHz) ↔ 전기적 거리 2D 히스토그램")
        fig_hd = px.density_heatmap(
            df_joint,
            x="distance",
            y="Nadir (mHz)",
            nbinsx=30,
            nbinsy=20,
            title="Disturbance vs Distance",
            color_continuous_scale=heatmap_scale
        )
        fig_hd.update_layout(
            xaxis_title="Electrical Distance (p.u. impedance)",
            yaxis_title="Frequency Nadir (mHz)",
            coloraxis_colorbar=dict(title="Count")
        )



        st.plotly_chart(fig_hd, use_container_width=True)
