"""
RL Visualization — Embedding Atlas
====================================
Three interactive sections powered by real RL training data:

  1. CartPole DQN  – Policy evolution with rich metadata and guided exploration
  2. Trajectories  – Follow individual episodes through the embedding
  3. LunarLander   – Dense rewards, 8D state, flight-phase clustering

Run with:
    marimo edit rl_visualization_tour.py

Prerequisites:
    pip install stable-baselines3 gymnasium umap-learn matplotlib
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", app_title="RL Visualization — Embedding Atlas")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell
def _():
    from embedding_atlas.widget import EmbeddingAtlasWidget
    from embedding_atlas.projection import compute_vector_projection
    import gymnasium as gym
    from stable_baselines3 import DQN
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return DQN, EmbeddingAtlasWidget, compute_vector_projection, gym, plt


@app.cell
def _():
    from embedding_atlas.rl_utils import (
        prepare_rl_data_for_projection,
        unpack_state_components,
        compute_td_error,
        format_state_description,
        compute_episode_metadata,
        plot_episode_timeline,
    )

    return (
        compute_episode_metadata,
        compute_td_error,
        format_state_description,
        plot_episode_timeline,
        prepare_rl_data_for_projection,
        unpack_state_components,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RL Replay Buffer Visualization

    This notebook demonstrates how **Embedding Atlas** can visualize reinforcement
    learning replay buffers — turning opaque training data into explorable,
    interactive maps.

    Inspired by [Vizarel (Deshpande & Schneider, 2020)](https://arxiv.org/abs/2007.05577),
    we project replay buffer experiences into 2D and enrich them with metadata that
    reveals **what the agent learned**, **where it struggled**, and **how its policy evolved**.

    | # | Section | What you'll see |
    |---|---------|-----------------|
    | 1 | **CartPole DQN** | Policy evolution — safe vs dangerous states, Q-value landscapes |
    | 2 | **Episode Trajectories** | Follow individual episodes through the embedding |
    | 3 | **LunarLander DQN** | Dense rewards, flight phases, landing vs crashing |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 1 · CartPole DQN — Watching an Agent Learn

    We train a DQN agent on **CartPole-v1** for **50 000 steps**. The agent starts
    with a random policy and gradually learns to balance the pole.

    The replay buffer captures every experience `(state, action, reward, next_state, done)`.
    We unpack the 4D state into named physical quantities, compute Q-values and
    TD errors, then project everything to 2D with UMAP.
    """)
    return


@app.cell(hide_code=True)
def _(DQN, gym, pd):
    """Train CartPole DQN for 50k steps and extract replay buffer."""
    TOTAL_STEPS = 50_000
    BUFFER_SIZE = TOTAL_STEPS + 1000

    env = gym.make("CartPole-v1")
    cartpole_model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=BUFFER_SIZE,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0,
        seed=42,
    )
    cartpole_model.learn(total_timesteps=TOTAL_STEPS)
    env.close()

    # Extract replay buffer
    _buf = cartpole_model.replay_buffer
    _n = _buf.pos

    _rows = [
        {
            "state": _buf.observations[i].flatten().tolist(),
            "action": int(_buf.actions[i].flat[0]),
            "reward": float(_buf.rewards[i].flat[0]),
            "next_state": _buf.next_observations[i].flatten().tolist(),
            "done": bool(_buf.dones[i].flat[0]),
            "timestep": i,
        }
        for i in range(_n)
    ]
    cartpole_df = pd.DataFrame(_rows)
    print(f"Extracted {len(cartpole_df)} replay buffer transitions.")
    return cartpole_df, cartpole_model


@app.cell(hide_code=True)
def _(
    cartpole_df,
    cartpole_model,
    compute_episode_metadata,
    compute_td_error,
    format_state_description,
    np,
    pd,
    unpack_state_components,
):
    """Build rich metadata for CartPole replay buffer."""
    if cartpole_df is not None:
        cp_df = cartpole_df.copy()

        # Unpack 4D state into named columns
        cp_df = unpack_state_components(
            cp_df,
            "state",
            [
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_angular_velocity",
            ],
        )

        # Human-readable action names
        cp_df["action_name"] = cp_df["action"].map({0: "push_left", 1: "push_right"})

        # Episode metadata
        cp_df = compute_episode_metadata(cp_df, done_col="done", reward_col="reward")

        # Training stage
        _n = len(cp_df)
        cp_df["training_stage"] = pd.cut(
            cp_df["timestep"],
            bins=[-1, _n // 3, 2 * _n // 3, _n],
            labels=["early", "mid", "late"],
        ).astype(str)

        # Pole danger categories
        cp_df["pole_danger"] = "safe"
        cp_df.loc[cp_df["pole_angle"].abs() > 0.10, "pole_danger"] = "warning"
        cp_df.loc[cp_df["pole_angle"].abs() > 0.15, "pole_danger"] = "critical"

        # Episode outcome
        cp_df["episode_outcome"] = np.where(
            cp_df["episode_length"] >= 200, "survived", "fell"
        )

        # Q-values from trained model
        import torch as _torch

        _q_vals, _max_qs = [], []
        with _torch.no_grad():
            for _, _r in cp_df.iterrows():
                _s = np.array(_r["state"], dtype=np.float32).reshape(1, -1)
                _ot, _ = cartpole_model.policy.obs_to_tensor(_s)
                _qa = cartpole_model.q_net(_ot).cpu().numpy()[0]
                _q_vals.append(float(_qa[int(_r["action"])]))
                _max_qs.append(float(np.max(_qa)))

        cp_df["q_value"] = _q_vals
        cp_df["max_q_value"] = _max_qs
        cp_df["advantage"] = cp_df["q_value"] - cp_df["max_q_value"]

        # Q-value categories
        _t33, _t67 = np.percentile(_q_vals, [33, 67])
        cp_df["q_category"] = pd.cut(
            cp_df["q_value"],
            bins=[-np.inf, _t33, _t67, np.inf],
            labels=["low_q", "mid_q", "high_q"],
        ).astype(str)

        # TD error
        cp_df = compute_td_error(cp_df, cartpole_model, gamma=0.99)
        _td_66, _td_90 = np.percentile(cp_df["td_error"], [66, 90])
        cp_df["td_category"] = "expected"
        cp_df.loc[cp_df["td_error"] > _td_66, "td_category"] = "surprising"
        cp_df.loc[cp_df["td_error"] > _td_90, "td_category"] = "very_surprising"

        # Skill level from episode length quartiles
        _ep_lens = cp_df["episode_length"]
        _q25, _q75 = np.percentile(_ep_lens[_ep_lens > 0], [25, 75])
        cp_df["skill_level"] = "intermediate"
        cp_df.loc[cp_df["episode_length"] <= _q25, "skill_level"] = "novice"
        cp_df.loc[cp_df["episode_length"] >= _q75, "skill_level"] = "expert"

        # Text description for hover
        cp_df["state_description"] = format_state_description(
            cp_df,
            component_cols=["cart_position", "pole_angle"],
            action_col="action_name",
            q_col="q_value",
            reward_col="reward",
        )

        cartpole_enriched = cp_df
        print(f"Enriched {len(cartpole_enriched)} transitions with metadata.")
    else:
        cartpole_enriched = None
        print("Skipped — data not available.")
    return


app._unparsable_cell(
    """
    \"\"\"UMAP projection of CartPole (state, action) vectors.\"\"\"

    cartpole_proj = cartpole_enriched.copy()
    cartpole_proj, _vec = prepare_rl_data_for_projection(
        cartpole_proj,
        observation_columns=[\"state\"],
        action_columns=[\"action\"],
    )
     compute_vector_projection(
        cartpole_proj, vector=_vec,
        x=\"projection_x\", y=\"projection_y\", neighbors=\"neighbors\",
        umap_args={\"n_neighbors\": 15, \"min_dist\": 0.1, \"metric\": \"cosine\"},
    )
    # Drop heavy columns not needed for visualization
    cartpole_proj = cartpole_proj.drop(
        columns=[c for c in [_vec, \"state\", \"next_state\"] if c in cartpole_proj.columns]
    )
    print(f\"CartPole projection ready: {len(cartpole_proj)} points\")
    """,
    column=None,
    disabled=False,
    hide_code=True,
    name="_",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What to look for

    - **Charts panel** (right side): Click on a bar in `pole_danger` to highlight
      *safe* vs *critical* states. Critical states (large pole angle) tend to cluster
      separately — these are moments right before the pole falls.
    - **`training_stage`**: Filter by *early* / *mid* / *late* to see how the agent's
      experience distribution shifts as it learns.
    - **`episode_outcome`**: *survived* episodes (length ≥ 200) cluster in the "safe"
      region of the embedding.
    - **Hover** over any point to see its state description (pole angle, cart position,
      action, Q-value).
    - **Lasso-select** a cluster to see detailed statistics below the widget.
    """)
    return


@app.cell
def _(EmbeddingAtlasWidget, cartpole_proj, mo):
    if cartpole_proj is not None:
        cartpole_widget = EmbeddingAtlasWidget(
            cartpole_proj,
            x="projection_x",
            y="projection_y",
            neighbors="neighbors",
            text="state_description",
            labels="automatic",
            show_charts=True,
            show_table=True,
            point_size=2.0,
        )
    else:
        cartpole_widget = None
        mo.md("Install SB3 + umap-learn to see this widget.")
    return (cartpole_widget,)


@app.cell
def _(cartpole_widget):
    cartpole_widget
    return


@app.cell
def _(cartpole_proj, cartpole_widget, mo, np, pd, plt):
    """Reactive selection analysis for CartPole."""
    if cartpole_widget is not None:
        _ = cartpole_widget._predicate
        _sel = cartpole_widget.selection()
    else:
        _sel = pd.DataFrame()

    if len(_sel) > 0:
        # Summary statistics
        _summary = mo.md(
            f"**Selection:** {len(_sel)} transitions"
            f" | training: {list(_sel['training_stage'].unique())}"
            f" | pole_danger: {list(_sel['pole_danger'].unique())}"
            f" | mean episode_length: **{_sel['episode_length'].mean():.0f}**"
            f" | mean Q: **{_sel['q_value'].mean():.2f}**"
            f" | mean TD error: **{_sel['td_error'].mean():.3f}**"
        )

        # Pole angle histogram: selection vs full dataset
        _fig1, _ax1 = plt.subplots(1, 1, figsize=(5, 3))
        _ax1.hist(
            cartpole_proj["pole_angle"], bins=40, alpha=0.4, label="All", density=True
        )
        _ax1.hist(
            _sel["pole_angle"], bins=40, alpha=0.7, label="Selected", density=True
        )
        _ax1.set_xlabel("Pole Angle (rad)")
        _ax1.set_ylabel("Density")
        _ax1.set_title("Pole Angle Distribution")
        _ax1.legend()
        _fig1.tight_layout()

        # Action distribution bar chart
        _fig2, _ax2 = plt.subplots(1, 1, figsize=(4, 3))
        _act_full = cartpole_proj["action_name"].value_counts(normalize=True)
        _act_sel = _sel["action_name"].value_counts(normalize=True)
        _actions = sorted(set(_act_full.index) | set(_act_sel.index))
        _x = np.arange(len(_actions))
        _ax2.bar(
            _x - 0.15,
            [_act_full.get(a, 0) for a in _actions],
            0.3,
            label="All",
            alpha=0.5,
        )
        _ax2.bar(
            _x + 0.15,
            [_act_sel.get(a, 0) for a in _actions],
            0.3,
            label="Selected",
            alpha=0.8,
        )
        _ax2.set_xticks(_x)
        _ax2.set_xticklabels(_actions)
        _ax2.set_ylabel("Proportion")
        _ax2.set_title("Action Distribution")
        _ax2.legend()
        _fig2.tight_layout()

        mo.vstack(
            [
                _summary,
                mo.hstack([mo.as_html(_fig1), mo.as_html(_fig2)]),
            ]
        )
        plt.close(_fig1)
        plt.close(_fig2)
    else:
        mo.md(
            "*No selection yet — **lasso a cluster** in the widget above to inspect it.*\n\n"
            "> **Quick guide:** Open the charts panel (sidebar icon). Click the `pole_danger: critical` "
            "bar to highlight dangerous states. Then lasso those points to see the analysis below.\n\n"
            "> The `td_category: very_surprising` filter highlights transitions where the agent's "
            "predictions were most wrong — these are the most informative experiences."
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 2 · Episode Trajectories — Following an Episode

    RL is fundamentally **sequential**. The UMAP projection above treats each
    transition as an independent point — losing the temporal structure.

    Here we restore that structure: **select an episode** with the slider and see:
    - Its **path through the embedding** (which regions of state-space the agent visited)
    - A **timeline** showing how the pole angle and Q-value evolved step by step

    > **Try this:** Compare an early episode (short, agent falls fast) with a late
    > episode (long, agent balances). Notice how late episodes trace extended paths
    > through the "safe" region of the embedding.
    """)
    return


@app.cell
def _(cartpole_proj, mo):
    """Episode selector slider."""
    if cartpole_proj is not None:
        _max_ep = int(cartpole_proj["episode"].max())
        # Pick a late episode as default (likely longer and more interesting)
        _default = max(0, _max_ep - 5)
        episode_slider = mo.ui.slider(
            start=0,
            stop=_max_ep,
            step=1,
            value=_default,
            label="Episode number",
        )
    else:
        episode_slider = None
    return (episode_slider,)


@app.cell
def _(episode_slider, mo):
    if episode_slider is not None:
        mo.hstack(
            [
                mo.md("**Select an episode:**"),
                episode_slider,
                mo.md(f"Episode **{episode_slider.value}**"),
            ]
        )
    return


@app.cell
def _(
    EmbeddingAtlasWidget,
    cartpole_proj,
    episode_slider,
    mo,
    plot_episode_timeline,
    plt,
):
    """Show episode trajectory widget and timeline side by side."""
    if cartpole_proj is not None and episode_slider is not None:
        _ep_num = episode_slider.value
        _ep_df = cartpole_proj[cartpole_proj["episode"] == _ep_num].copy()

        if len(_ep_df) > 0:
            # Episode info
            _ep_info = mo.md(
                f"**Episode {_ep_num}:** {len(_ep_df)} steps"
                f" | outcome: **{_ep_df['episode_outcome'].iloc[0]}**"
                f" | skill: **{_ep_df['skill_level'].iloc[0]}**"
                f" | training stage: **{_ep_df['training_stage'].iloc[0]}**"
                f" | mean Q: **{_ep_df['q_value'].mean():.2f}**"
            )

            # Widget showing just this episode's points
            _ep_widget = EmbeddingAtlasWidget(
                _ep_df,
                x="projection_x",
                y="projection_y",
                neighbors="neighbors",
                text="state_description",
                show_charts=False,
                show_table=True,
                point_size=4.0,
            )

            # Timeline plot
            _fig = plot_episode_timeline(
                cartpole_proj,
                _ep_num,
                state_components=["pole_angle", "cart_position"],
                q_col="q_value",
                action_col="action_name",
            )

            mo.vstack(
                [
                    _ep_info,
                    mo.hstack([_ep_widget, mo.as_html(_fig)]),
                ]
            )
            plt.close(_fig)
        else:
            mo.md(f"Episode {_ep_num} has no data.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 3 · LunarLander DQN — Dense Rewards and Rich State

    LunarLander provides **dense reward shaping**: the agent earns continuous rewards
    for approaching the landing pad and penalties for using fuel, plus large bonuses
    (+100 for landing, -100 for crashing).

    The 8D state captures intuitive physics: position, velocity, angle, angular
    velocity, and leg contact. We unpack these into named columns so you can
    **see clusters correspond to flight phases** — high altitude, approach, and landing.

    | Column | Meaning |
    |--------|---------|
    | `flight_phase` | high_altitude / approach / landing (based on altitude) |
    | `action_name` | noop / left_engine / main_engine / right_engine |
    | `reward_category` | high_penalty / small_penalty / small_reward / high_reward |
    | `landing_outcome` | landed / crashed / in_flight (terminal states only) |
    | `speed` | sqrt(vel_x^2 + vel_y^2) — how fast the lander is moving |
    """)
    return


@app.cell(hide_code=True)
def _(DQN, gym, pd):
    """Train LunarLander DQN for 100k steps and extract replay buffer."""
    LUNAR_STEPS = 200_000
    LUNAR_BUFFER = LUNAR_STEPS + 2000

    lunar_env = gym.make("LunarLander-v3")
    lunar_model = DQN(
        "MlpPolicy",
        lunar_env,
        learning_rate=5e-4,
        buffer_size=LUNAR_BUFFER,
        learning_starts=2000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=500,
        verbose=0,
        seed=42,
    )
    lunar_model.learn(total_timesteps=LUNAR_STEPS)
    lunar_env.close()

    _buf = lunar_model.replay_buffer
    _n = _buf.pos

    _rows = [
        {
            "state": _buf.observations[i].flatten().tolist(),
            "action": int(_buf.actions[i].flat[0]),
            "reward": float(_buf.rewards[i].flat[0]),
            "next_state": _buf.next_observations[i].flatten().tolist(),
            "done": bool(_buf.dones[i].flat[0]),
            "timestep": i,
        }
        for i in range(_n)
    ]
    lunar_df = pd.DataFrame(_rows)
    print(f"Extracted {len(lunar_df)} LunarLander transitions.")

    return lunar_df, lunar_model


@app.cell(hide_code=True)
def _(
    compute_episode_metadata,
    compute_td_error,
    format_state_description,
    lunar_df,
    lunar_model,
    np,
    pd,
    unpack_state_components,
):
    """Build rich metadata for LunarLander replay buffer."""
    if lunar_df is not None:
        ll_df = lunar_df.copy()

        # Unpack 8D state
        ll_df = unpack_state_components(
            ll_df,
            "state",
            [
                "pos_x",
                "pos_y",
                "vel_x",
                "vel_y",
                "angle",
                "angular_vel",
                "left_leg",
                "right_leg",
            ],
        )

        # Derived physics columns
        ll_df["altitude"] = ll_df["pos_y"]
        ll_df["speed"] = np.sqrt(ll_df["vel_x"] ** 2 + ll_df["vel_y"] ** 2)

        # Action names
        LUNAR_ACTIONS = {
            0: "noop",
            1: "left_engine",
            2: "main_engine",
            3: "right_engine",
        }
        ll_df["action_name"] = ll_df["action"].map(LUNAR_ACTIONS)

        # Episode metadata
        ll_df = compute_episode_metadata(ll_df, done_col="done", reward_col="reward")

        # Training stage
        _n = len(ll_df)
        ll_df["training_stage"] = pd.cut(
            ll_df["timestep"],
            bins=[-1, _n // 3, 2 * _n // 3, _n],
            labels=["early", "mid", "late"],
        ).astype(str)

        # Flight phase based on altitude
        ll_df["flight_phase"] = "approach"
        ll_df.loc[ll_df["pos_y"] > 0.5, "flight_phase"] = "high_altitude"
        ll_df.loc[ll_df["pos_y"] < 0.1, "flight_phase"] = "landing"

        # Reward categories
        ll_df["reward_category"] = pd.cut(
            ll_df["reward"],
            bins=[-np.inf, -0.5, 0, 0.5, np.inf],
            labels=["high_penalty", "small_penalty", "small_reward", "high_reward"],
        ).astype(str)

        # Landing outcome (for terminal states)
        ll_df["landing_outcome"] = "in_flight"
        _terminal = ll_df["done"] == True
        ll_df.loc[_terminal & (ll_df["reward"] > 50), "landing_outcome"] = "landed"
        ll_df.loc[_terminal & (ll_df["reward"] < -50), "landing_outcome"] = "crashed"

        # Q-values
        import torch as _torch

        _q_vals = []
        with _torch.no_grad():
            for _, _r in ll_df.iterrows():
                _s = np.array(_r["state"], dtype=np.float32).reshape(1, -1)
                _ot, _ = lunar_model.policy.obs_to_tensor(_s)
                _qa = lunar_model.q_net(_ot).cpu().numpy()[0]
                _q_vals.append(float(_qa[int(_r["action"])]))
        ll_df["q_value"] = _q_vals

        # TD error
        ll_df = compute_td_error(ll_df, lunar_model, gamma=0.99)

        # Text description for hover
        ll_df["state_description"] = format_state_description(
            ll_df,
            component_cols=["pos_x", "pos_y", "speed", "angle"],
            action_col="action_name",
            q_col="q_value",
            reward_col="reward",
        )

        lunar_enriched = ll_df
        print(f"Enriched {len(lunar_enriched)} LunarLander transitions.")
    else:
        lunar_enriched = None
        print("Skipped — data not available.")
    return (lunar_enriched,)


@app.cell(hide_code=True)
def _(
    compute_vector_projection,
    lunar_enriched,
    prepare_rl_data_for_projection,
):
    """UMAP projection of LunarLander (state, action) vectors."""
    lunar_proj = lunar_enriched.copy()
    lunar_proj, _vec = prepare_rl_data_for_projection(
        lunar_proj,
        observation_columns=["state"],
        action_columns=["action"],
    )
    compute_vector_projection(
        lunar_proj,
        vector=_vec,
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        umap_args={"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
    )

    # Add caching

    lunar_proj = lunar_proj.drop(
        columns=[c for c in [_vec, "state", "next_state"] if c in lunar_proj.columns]
    )
    print(f"LunarLander projection ready: {len(lunar_proj)} points")
    return (lunar_proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What to look for

    - **`flight_phase`** in the charts panel creates the most visible clustering:
      *high_altitude* points (early in episode) cluster separately from *landing*
      points (near the pad). Filter each phase to see how they map spatially.
    - **`reward_category`**: *high_reward* points cluster near successful landings.
      *high_penalty* points cluster near crashes and excessive fuel use.
    - **`landing_outcome`**: Filter to *landed* or *crashed* to see where successful
      vs failed episodes end up in the embedding.
    - **`training_stage`**: Compare *early* (random behavior) vs *late* (learned policy)
      to see how the experience distribution shifts toward successful regions.
    - **Hover** shows position, speed, angle, action, Q-value, and reward.
    """)
    return


@app.cell
def _(EmbeddingAtlasWidget, lunar_proj):
    lunar_widget = EmbeddingAtlasWidget(
        lunar_proj,
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        text="state_description",
        labels="automatic",
        show_charts=True,
        show_table=True,
        point_size=2.0,
    )
    return (lunar_widget,)


@app.cell
def _(lunar_widget):
    lunar_widget
    return


if __name__ == "__main__":
    app.run()
