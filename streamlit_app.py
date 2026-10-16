import io
import time

import streamlit as st
from PIL import Image

from agent import Agent, train_one_step
from game import SnakeGameAI

st.set_page_config(
    page_title="Snake AI Lab",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: rgba(120,130,150,.08); padding: 14px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🐍 Snake AI Lab")
st.caption("Deep Q-Learning · Train live in your browser · Model weights are not saved to disk")

# Initialize session state once per browser session.
if "agent" not in st.session_state:
    st.session_state.agent = Agent()
    st.session_state.game = SnakeGameAI()
    st.session_state.training = True
    st.session_state.started_at = time.monotonic()
    st.session_state.elapsed_before_pause = 0.0
    st.session_state.scores = []
    st.session_state.steps = 0
    st.session_state.best_score = 0

with st.sidebar:
    st.subheader("Training controls")
    speed = st.select_slider(
        "Display refresh interval",
        options=["Fast", "Balanced", "Relaxed"],
        value="Relaxed",
        help="A slower refresh is less demanding on Streamlit Cloud and reduces visible flicker.",
    )
    # Streamlit fragments update only the live panel, not the whole page.
    refresh_seconds = {"Fast": 0.35, "Balanced": 0.6, "Relaxed": 1.0}[speed]

    c1, c2 = st.columns(2)
    if c1.button("▶ Start", use_container_width=True, type="primary"):
        if not st.session_state.training:
            st.session_state.training = True
            if st.session_state.started_at is None:
                st.session_state.started_at = time.monotonic()

    if c2.button("⏸ Pause", use_container_width=True):
        if st.session_state.training:
            st.session_state.training = False
            if st.session_state.started_at is not None:
                st.session_state.elapsed_before_pause += (
                    time.monotonic() - st.session_state.started_at
                )
                st.session_state.started_at = None

    if st.button("↻ Reset training", use_container_width=True):
        st.session_state.agent = Agent()
        st.session_state.game = SnakeGameAI()
        st.session_state.training = True
        st.session_state.started_at = time.monotonic()
        st.session_state.elapsed_before_pause = 0.0
        st.session_state.scores = []
        st.session_state.steps = 0
        st.session_state.best_score = 0
        st.rerun()

    st.divider()
    st.caption(
        "Training runs only while this app session is active. "
        "Resetting creates a fresh, untrained agent."
    )

# Create all output slots once, outside the fragment. The fragment only updates
# their contents, rather than rebuilding the image/chart layout each refresh.
metric_cols = st.columns(4)
elapsed_slot = metric_cols[0].empty()
episode_slot = metric_cols[1].empty()
best_slot = metric_cols[2].empty()
current_slot = metric_cols[3].empty()

left_col, right_col = st.columns([1.1, 1], gap="large")
with left_col:
    st.subheader("Live game")
    game_image_slot = st.empty()
    game_status_slot = st.empty()

with right_col:
    st.subheader("Learning progress")
    graph_slot = st.empty()
    graph_caption_slot = st.empty()

about_slot = st.container()


def render_graph(scores):
    if not scores:
        graph_slot.info(
            "Training is running. The graph will show its first point when an episode finishes."
        )
        graph_caption_slot.empty()
        return

    averages = [
        sum(scores[max(0, i - 19):i + 1]) / len(scores[max(0, i - 19):i + 1])
        for i in range(len(scores))
    ]

    width, height = 600, 330
    pad_left, pad_right, pad_top, pad_bottom = 42, 18, 35, 35
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom
    max_score = max(1.0, float(max(scores)), max(averages)) * 1.15

    def get_points(values):
        points = []
        denominator = max(1, len(values) - 1)
        for i, value in enumerate(values):
            x = pad_left + i * plot_width / denominator
            y = pad_top + plot_height - (float(value) / max_score) * plot_height
            points.append(f"{x:.1f},{y:.1f}")
        return " ".join(points)

    grid_lines = []
    for tick in range(5):
        y = pad_top + tick * plot_height / 4
        value = max_score * (1 - tick / 4)
        grid_lines.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" '
            f'x2="{width-pad_right}" y2="{y:.1f}" '
            f'stroke="#334155" stroke-width="1"/>'
            f'<text x="{pad_left-8}" y="{y+4:.1f}" text-anchor="end" '
            f'fill="#CBD5E1" font-size="11">{value:.1f}</text>'
        )

    svg = f"""
    <div style="background:#111827;border-radius:12px;padding:10px 10px 14px;">
      <svg viewBox="0 0 {width} {height}" width="100%" role="img"
           aria-label="Snake AI training progress graph">
        <text x="{pad_left}" y="20" fill="#F1F5F9" font-size="13">Score per episode</text>
        {''.join(grid_lines)}
        <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height-pad_bottom}"
              stroke="#64748B" stroke-width="1"/>
        <line x1="{pad_left}" y1="{height-pad_bottom}" x2="{width-pad_right}" y2="{height-pad_bottom}"
              stroke="#64748B" stroke-width="1"/>
        <polyline points="{get_points(scores)}" fill="none" stroke="#38BDF8"
                  stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
        <polyline points="{get_points(averages)}" fill="none" stroke="#F59E0B"
                  stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
        <text x="{pad_left}" y="{height-10}" fill="#CBD5E1" font-size="11">Episode 1</text>
        <text x="{width-pad_right}" y="{height-10}" text-anchor="end"
              fill="#CBD5E1" font-size="11">Episode {len(scores)}</text>
      </svg>
      <div style="display:flex;gap:18px;flex-wrap:wrap;color:#E5E7EB;
                  font-size:13px;padding-left:30px;">
        <span style="color:#38BDF8">● Episode score</span>
        <span style="color:#F59E0B">● Rolling average (up to 20)</span>
      </div>
    </div>
    """
    graph_slot.markdown(svg, unsafe_allow_html=True)
    graph_caption_slot.caption(
        "The graph updates when an episode ends. The average uses up to the latest 20 episodes."
    )


@st.fragment(run_every=refresh_seconds if st.session_state.training else None)
def live_training_panel():
    agent = st.session_state.agent
    game = st.session_state.game

    if st.session_state.training:
        # Advance several moves per refresh to reduce UI overhead while keeping
        # the game animation responsive. Stop the batch if an episode ends.
        for _ in range(8):
            result = train_one_step(agent, game)
            st.session_state.steps += 1
            if result["done"]:
                score = int(result["score"])
                st.session_state.scores.append(score)
                st.session_state.best_score = max(st.session_state.best_score, score)
                break

    elapsed = st.session_state.elapsed_before_pause
    if st.session_state.started_at is not None:
        elapsed += time.monotonic() - st.session_state.started_at
    elapsed_int = int(elapsed)
    timer_text = (
        f"{elapsed_int // 3600:02d}:"
        f"{(elapsed_int % 3600) // 60:02d}:"
        f"{elapsed_int % 60:02d}"
    )

    elapsed_slot.metric("Elapsed time", timer_text)
    episode_slot.metric("Episode", f"{agent.n_games:,}")
    best_slot.metric("Best score", st.session_state.best_score)
    current_slot.metric("Current score", game.score)

    # Encode an RGB frame to PNG bytes. The stable placeholder is reused every tick.
    frame = Image.fromarray(game.render().astype("uint8"), mode="RGB")
    buffer = io.BytesIO()
    frame.save(buffer, format="PNG", optimize=False)
    game_image_slot.image(buffer.getvalue(), use_container_width=True)
    status = "🟢 Training" if st.session_state.training else "⏸ Paused"
    game_status_slot.caption(
        f"{status} · Moves processed: {st.session_state.steps:,}"
    )

    render_graph(st.session_state.scores)


live_training_panel()

with about_slot:
    with st.expander("About this training run"):
        st.write(
            "The agent uses an 11-value state, a 256-unit hidden layer, three possible actions, "
            "experience replay, and a discount factor of 0.9. Model weights exist in memory only; "
            "this app does not save a .pth file."
        )
