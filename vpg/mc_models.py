from rlcard import make as make_env
from agents import EveryVisitMCAgent, RandomAgent
from experiments import play_episodes, evaluate_agents


def train_and_save_mc_agents(
    env,
    state_transformer,
    prefix,
    epsilon=0.01,
    gamma_aggro=0.1,
    gamma_std=0.99,
    num_train_episodes=1_000_000,
    update_freq=100,
    num_eval_episodes=10_000,
    save_dir="agents",
):
    """
    Train two EveryVisitMCAgents (aggro & std) vs RandomAgent, evaluate them,
    and save their models to disk.

    Args:
        env:           an rlcard.Env instance
        state_transformer:  function mapping raw state -> MC state
        prefix:        str label to prepend to saved filenames (e.g. 'gen0_v2l')
        epsilon:       exploration rate for both agents
        gamma_aggro:   discount factor for the 'aggressive' agent
        gamma_std:     discount factor for the 'standard' agent
        num_train_episodes: total self‐play episodes per training run
        update_freq:   how often to update the MC estimates
        num_eval_episodes: episodes per evaluation
        save_dir:      directory to write the .pkl files into
    """
    random_agent = RandomAgent()

    # instantiate agents
    aggro = EveryVisitMCAgent(
        epsilon=epsilon,
        gamma=gamma_aggro,
        state_transformer=state_transformer,
    )
    std = EveryVisitMCAgent(
        epsilon=epsilon,
        gamma=gamma_std,
        state_transformer=state_transformer,
    )

    # --- Training ---
    print(f"Training {prefix}_aggro (γ={gamma_aggro}) vs. random agent")
    play_episodes(
        env,
        aggro,
        random_agent,
        num_episodes=num_train_episodes,
        do_update=True,
        update_freq=update_freq,
        state_transformer=state_transformer,
    )

    print(f"Training random agent vs. {prefix}_std (γ={gamma_std})")
    play_episodes(
        env,
        random_agent,
        std,
        num_episodes=num_train_episodes,
        do_update=True,
        update_freq=update_freq,
        state_transformer=state_transformer,
    )

    # --- Evaluation ---
    agg_reward, rand_reward1 = evaluate_agents(
        env, aggro, random_agent, num_eval_episodes
    )
    rand_reward2, std_reward = evaluate_agents(
        env, random_agent, std, num_eval_episodes
    )

    print(
        f"Eval payouts {prefix}_aggro vs random: {agg_reward:.3f}, {rand_reward1:.3f}"
    )
    print(
        f"Eval payouts {prefix}_std   vs random: {std_reward:.3f}, {rand_reward2:.3f}"
    )

    # --- Saving ---
    aggro_path = f"{save_dir}/{prefix}_aggro.pkl"
    std_path = f"{save_dir}/{prefix}_std.pkl"
    aggro.save(aggro_path)
    std.save(std_path)
    print(f"Saved models to:\n  • {aggro_path}\n  • {std_path}")


env = make_env("limit-holdem")
from utils import process_limit_state_v2, process_limit_state_v1

train_and_save_mc_agents(
    env=env,
    state_transformer=process_limit_state_v1,
    prefix="gen0_v1l",
    epsilon=0.01,
    gamma_aggro=0.1,
    gamma_std=0.99,
    num_train_episodes=1_000_000,
    update_freq=100,
    num_eval_episodes=10_000,
    save_dir="agents",
)

# train_and_save_mc_agents(
#     env=env,
#     state_transformer=process_limit_state_v2,
#     prefix="gen0_v1l",
#     epsilon=0.01,
#     gamma_aggro=0.1,
#     gamma_std=0.99,
#     num_train_episodes=1_000_000,
#     update_freq=100,
#     num_eval_episodes=10_000,
#     save_dir="agents",
# )

# save =============================================================================================

# agent_gen0_aggro_v2l = EveryVisitMCAgent(
#     epsilon=EPSILON, gamma=0.1, state_transformer=process_limit_state_v2
# )
# agent_gen0_std_v2l = EveryVisitMCAgent(
#     epsilon=EPSILON, gamma=0.99, state_transformer=process_limit_state_v2
# )

# print("Training agent_gen0_aggro_v2l vs. random agent")
# train_payoffs = play_episodes(
#     env,
#     agent_gen0_aggro_v2l,
#     random_agent,
#     num_episodes=NUM_EPISODES,
#     do_update=True,
#     update_freq=UPDATE_FREQ,
#     state_transformer=process_limit_state_v2,
# )

# print("Training agent_gen0_std_v2l vs. random agent")
# train_payoffs = play_episodes(
#     env,
#     random_agent,
#     agent_gen0_std_v2l,
#     num_episodes=NUM_EPISODES,
#     do_update=True,
#     update_freq=UPDATE_FREQ,
#     state_transformer=process_limit_state_v2,
# )

# g0av2l, r_g0av2l = evaluate_agents(
#     env, agent_gen0_aggro_v2l, random_agent, num_episodes=10000
# )
# r_g0sv2l, g0sv2l = evaluate_agents(
#     env, random_agent, agent_gen0_std_v2l, num_episodes=10000
# )

# print("Eval payouts g0av2l vs random:", g0av2l, r_g0av2l)
# print("Eval payouts g0sv2l vs random:", g0sv2l, r_g0sv2l)

# print("Saving gen 0 v2l agents")
# agent_gen0_aggro_v2l.save("agents/agent_gen0_aggro_v2l.pkl")
# agent_gen0_std_v2l.save("agents/agent_gen0_std_v2l.pkl")
