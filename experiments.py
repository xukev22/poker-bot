import random

from utils import process_leduc_state_v1
import matplotlib.pyplot as plt


def play_episodes(
    env,
    agent0,
    agent1,
    num_episodes=1000,
    do_update=True,
    update_freq=1,
    state_transformer=process_leduc_state_v1,
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
                info_s = state_transformer(state_for_pid, pid)

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


def evaluate_agents(env, agent0, agent1, num_episodes=1000, plot=False):
    """
    Plays `num_episodes` episodes of agent0 vs. agent1 and returns
    the average payoff of (player0, player1).

    If plot=True, displays a simple line chart of each player's
    rewards across episodes.
    """
    # Gather payoffs from each episode
    payoffs = play_episodes(env, agent0, agent1, num_episodes, do_update=False)
    # Separate payoffs for each agent
    p0_rewards = [p[0] for p in payoffs]
    p1_rewards = [p[1] for p in payoffs]

    # Compute averages
    avg_p0 = sum(p0_rewards) / num_episodes
    avg_p1 = sum(p1_rewards) / num_episodes

    # Optionally plot
    if plot:
        plt.figure()
        plt.plot(p0_rewards, label="Player 0 Reward")
        plt.plot(p1_rewards, label="Player 1 Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.title("Rewards Over Time")
        plt.show()

    return (avg_p0, avg_p1)


def flip(pid):
    if pid == 0:
        return 1
    if pid == 1:
        return 0
    raise RuntimeError("Flip pid should be 0 or 1")


def human_play_bot(env, human_agent, bot_agent, human_first):
    while True:
        env.reset()
        print("*" * 40)
        print("new game!")
        print(
            "dont look here if you dont want to cheat!",
            env.get_perfect_information(),
        )
        init_pid = env.get_player_id()
        print(init_pid, "init pid <-------------------")
        human_pid = init_pid if human_first else flip(init_pid)

        while not env.is_over():
            pid = env.get_player_id()
            state_for_pid = env.get_state(pid)

            if pid == human_pid:
                action = human_agent.step(state_for_pid)
                print("*" * 20)
                print("you took action:", action)
            else:
                action = bot_agent.step(state_for_pid, True)
                print("bot took action:", action)

            # step the env
            env.step(action, True)

        # get final payoffs
        payoffs = env.get_payoffs()  # [payoff_p0, payoff_p1]
        print(payoffs[human_pid])
