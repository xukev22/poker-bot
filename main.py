import rlcard

from agents import FirstVisitMCAgent, RandomAgent, HumanAgent
from utils import process_leduc_state_v1


def play_episodes(
    env, agent0, agent1, num_episodes=1000, do_update=True, update_freq=1
):
    """
    Run 'num_episodes' episodes of the environment with agent0 (player 0) and agent1 (player 1).

    Arguments:
        env: An RLCard environment (e.g. rlcard.make("leduc-holdem")).
        agent0: Agent controlling player 0
        agent1: Agent controlling player 1
        num_episodes (int): How many episodes to run.
        do_update (bool): If True, we collect trajectories and update the agents.
        update_freq (int): How often (in number of episodes) to call .update().
                           E.g. update_freq=10 means we call .update() every 10 episodes.

    Returns:
        payoffs_history (list): A list of [payoff_p0, payoff_p1] for each episode.
    """

    payoffs_history = []  # store final payoffs of each episode

    # we need to store all episode trajectories:
    # a separate buffer for each agent, so we can do batch updates later
    # or we store them episode by episode if do_update=True
    if do_update:
        all_trajectories_0 = []
        all_trajectories_1 = []

    for episode_id in range(num_episodes):
        env.reset()

        # we collect transitions for each agent:
        episode_traj_0 = []  # (state, action, reward)
        episode_traj_1 = []

        while not env.is_over():
            pid = env.get_player_id()
            state_for_pid = env.get_state(pid)

            if pid == 0:
                action = agent0.step(state_for_pid)
            else:
                action = agent1.step(state_for_pid)

            # if learning with MC, store (state, action, reward=0) for the current pid
            if do_update:
                # minimal representation – or direct raw state – whichever you prefer
                # assume we have a function process_leduc_state_v1, TODO might parametrize, maybe store raw state?
                info_s = process_leduc_state_v1(state_for_pid, pid)

                if pid == 0:
                    episode_traj_0.append((info_s, action, 0.0))
                else:
                    episode_traj_1.append((info_s, action, 0.0))

            # step the env
            env.step(action, True)

        # get final payoffs
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

            # store these new episodes in a bigger buffer
            all_trajectories_0.append(episode_traj_0)
            all_trajectories_1.append(episode_traj_1)

            # update every `update_freq` episodes
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


def human_play_bot(env, human_agent, bot_agent):
    env.reset()
    print("*" * 40)
    print("new game!")
    while not env.is_over():
        pid = env.get_player_id()
        state_for_pid = env.get_state(pid)

        if pid == 0:
            action = human_agent.step(state_for_pid)
            print("*" * 20)
            print("you took action:", action)
        else:
            action = bot_agent.step(state_for_pid)
            print("bot took action:", action)

        # step the env
        env.step(action, True)

    # get final payoffs
    payoffs = env.get_payoffs()  # [payoff_p0, payoff_p1]
    print(payoffs)


env_human = rlcard.make("leduc-holdem")
human_agent = HumanAgent()
human_play_bot(env_human, human_agent, agent0)
