import rlcard

# from rlcard.agents import RandomAgent
# from rlcard.agents.human_agents.leduc_holdem_human_agent import HumanAgent

from agents import FirstVisitMCAgent, RandomAgent
from utils import process_leduc_state_v1


def play_episodes(
    env, agent0, agent1, num_episodes=1000, do_update=True, update_freq=1
):
    """
    Run 'num_episodes' episodes of the environment with agent0 (player 0) and agent1 (player 1).

    Arguments:
        env: An RLCard environment (e.g. rlcard.make("leduc-holdem")).
        agent0: Agent controlling player 0 (must have .step(state) method; optional .update()).
        agent1: Agent controlling player 1.
        num_episodes (int): How many episodes to run.
        do_update (bool): If True, we collect trajectories and update the agents.
        update_freq (int): How often (in number of episodes) to call .update().
                           E.g. update_freq=10 means we call .update() every 10 episodes.

    Returns:
        payoffs_history (list): A list of [payoff_p0, payoff_p1] for each episode.
    """

    payoffs_history = []  # store final payoffs of each episode

    # If we want to do MC learning, we need to store all episode trajectories:
    # a separate buffer for each agent, so we can do batch updates later
    # or we store them episode by episode if do_update=True
    if do_update:
        all_trajectories_0 = []
        all_trajectories_1 = []

    for episode_id in range(num_episodes):
        env.reset()

        # We'll collect transitions for each agent:
        episode_traj_0 = []  # (state, action, reward)
        episode_traj_1 = []

        while not env.is_over():
            pid = env.get_player_id()
            state_for_pid = env.get_state(pid)

            if pid == 0:
                action = agent0.step(state_for_pid)
            else:
                action = agent1.step(state_for_pid)

            # If we’re learning with MC, store (state, action, reward=0) for the current pid
            if do_update:
                # Minimal representation – or direct raw state – whichever you prefer
                # But let's assume we have a function process_leduc_state_v1
                # For general usage, you can store the raw state if your agent is used to that

                info_s = process_leduc_state_v1(state_for_pid, pid)

                if pid == 0:
                    episode_traj_0.append((info_s, action, 0.0))
                else:
                    episode_traj_1.append((info_s, action, 0.0))

            # Step the env
            env.step(action, True)

        # Episode is over, get final payoffs
        payoffs = env.get_payoffs()  # [payoff_p0, payoff_p1]
        payoffs_history.append(payoffs)

        if do_update:
            # Overwrite last transition's reward
            if episode_traj_0:
                s_last, a_last, _ = episode_traj_0[-1]
                episode_traj_0[-1] = (s_last, a_last, payoffs[0])

            if episode_traj_1:
                s_last, a_last, _ = episode_traj_1[-1]
                episode_traj_1[-1] = (s_last, a_last, payoffs[1])

            # Store these new episodes in a bigger buffer
            all_trajectories_0.append(episode_traj_0)
            all_trajectories_1.append(episode_traj_1)

            # Now either update right away or only every `update_freq` episodes
            if (episode_id + 1) % update_freq == 0:
                # If your agent has .update() which accepts a list of episodes
                # We feed all the new ones we've collected since last update:
                agent0.update(all_trajectories_0)
                agent1.update(all_trajectories_1)

                # Clear the buffer for next batch
                all_trajectories_0 = []
                all_trajectories_1 = []

    return payoffs_history


def evaluate_agents(env, agent0, agent1, num_episodes=1000):
    """
    Simple evaluation of agent0 vs agent1 over `num_episodes`.
    Returns the average payoff of (player0, player1).
    """
    payoffs = play_episodes(env, agent0, agent1, num_episodes, do_update=False)
    # payoffs is a list of [p0, p1] pairs
    avg_p0 = sum([p[0] for p in payoffs]) / num_episodes
    avg_p1 = sum([p[1] for p in payoffs]) / num_episodes
    return (avg_p0, avg_p1)


env = rlcard.make("leduc-holdem")
agent0 = FirstVisitMCAgent(epsilon=0.1, gamma=0.9)
agent1 = FirstVisitMCAgent(epsilon=0.1, gamma=0.9)

print(
    "Training the two MC agents vs each other for 10,000 episodes, updating every 20 episodes..."
)
train_payoffs = play_episodes(
    env, agent0, agent1, num_episodes=10000, do_update=True, update_freq=20
)

env_eval = rlcard.make("leduc-holdem")
agent_random = RandomAgent()

avg_return0, avg_return1 = evaluate_agents(
    env_eval, agent0, agent_random, num_episodes=10000
)
print(
    f"Trained agent0 vs Random -> avg payoff: {avg_return0}, random payoff: {avg_return1}"
)

avg_return0, avg_return1 = evaluate_agents(env_eval, agent0, agent1, num_episodes=10000)
print(
    f"Trained agent0 vs Random -> avg payoff: {avg_return0}, random payoff: {avg_return1}"
)

# env = rlcard.make("leduc-holdem")

# fv_mc_0 = FirstVisitMCAgent(epsilon=0.1, gamma=0.9)
# fv_mc_1 = FirstVisitMCAgent(epsilon=0.1, gamma=0.9)

# NUM_EPISODES = 10000

# for episode_id in range(NUM_EPISODES):
#     # Reset the environment at the start of the episode
#     env.reset()

#     # Keep separate trajectories for each player
#     trajectories_0 = []
#     trajectories_1 = []

#     while not env.is_over():
#         pid = env.get_player_id()
#         print(pid, "pid")
#         state_for_pid = env.get_state(pid)
#         print(state_for_pid, "state_for_pid")

#         # Decide an action via each agent’s step()
#         next_act = None
#         if pid == 0:
#             next_act = fv_mc_0.step(state_for_pid)
#         elif pid == 1:
#             next_act = fv_mc_1.step(state_for_pid)
#         else:
#             raise RuntimeError("PID should be 0 or 1 in heads-up!")

#         # Store (state, action, reward=0) in that player’s trajectory
#         # We only get the "imperfect" raw state for pid here;
#         # process_leduc_state_v1() extracts the minimal representation.
#         info_s = process_leduc_state_v1(state_for_pid, pid)

#         if pid == 0:
#             trajectories_0.append((info_s, next_act, 0.0))
#         else:
#             trajectories_1.append((info_s, next_act, 0.0))

#         # Apply the action
#         print(next_act, "next_act")
#         state = env.step(next_act, True)

#     # Now the episode is over, retrieve final payoffs
#     payoffs = env.get_payoffs()
#     print(payoffs, "payoffs\n\n\n\n\n")

#     # Update reward on the final transition of each trajectory
#     if trajectories_0:
#         # Replace the last (s,a,0) with (s,a,payoffs[0])
#         s_last, a_last, _ = trajectories_0[-1]
#         trajectories_0[-1] = (s_last, a_last, payoffs[0])

#     if trajectories_1:
#         s_last, a_last, _ = trajectories_1[-1]
#         trajectories_1[-1] = (s_last, a_last, payoffs[1])

#     # Perform First‑Visit MC updates for each agent using its own trajectory
#     # Each agent’s update() expects a list of “episodes,” so we wrap the single
#     # trajectory in a list: [trajectories_0].
#     fv_mc_0.update([trajectories_0])
#     fv_mc_1.update([trajectories_1])

#     print(f"Episode {episode_id+1} done. Payoffs = {payoffs}\n")


# env_eval = rlcard.make("leduc-holdem")

# # Create the random agent
# random_agent_1 = RandomAgent()

# NUM_EVAL_EPISODES = 5000
# cumulative_payoff_0 = 0.0

# for i in range(NUM_EVAL_EPISODES):
#     env_eval.reset()

#     while not env_eval.is_over():
#         pid = env_eval.get_player_id()
#         state_for_pid = env_eval.get_state(pid)

#         if pid == 0:
#             # Trained agent picks an action
#             action = fv_mc_0.step(state_for_pid)
#         else:
#             # Random agent picks an action
#             action = random_agent_1.step(state_for_pid)

#         env_eval.step(action, True)

#     payoffs = env_eval.get_payoffs()  # [ payoff_player0, payoff_player1 ]
#     cumulative_payoff_0 += payoffs[0]

# # Print the average payoff (or “win rate” if you treat >0 as a “win”)
# avg_payoff_0 = cumulative_payoff_0 / NUM_EVAL_EPISODES
# print(
#     f"After {NUM_EVAL_EPISODES} episodes vs. RandomAgent, MC agent average payoff is {avg_payoff_0:.3f}"
# )
