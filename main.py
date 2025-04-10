import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents.human_agents.leduc_holdem_human_agent import HumanAgent

from agents import FirstVisitMCAgent

env = rlcard.make("leduc-holdem")

human_agent = HumanAgent(num_actions=env.num_actions)
random_agent = RandomAgent(num_actions=env.num_actions)
random_agent2 = RandomAgent(num_actions=env.num_actions)

# env.set_agents([random_agent2, random_agent])
env.reset()
# for i in range(5):
#     trajectories, payoffs = env.run(is_training=False)
#     # print(env.get_perfect_information())
#     # print(env.get_payoffs())
#     print(trajectories, "\n\n\n\n\n")
#     # print("Payoffs:", payoffs)

print(env.get_perfect_information())

print(env.is_over())

test = FirstVisitMCAgent()

playerid = env.get_player_id()
print(playerid, "playerid")
state = env.get_state(1)
print(test.step(state), "YAYAYYA")
print(state, "state for pid1")
env.step("fold", True)
print(env.get_payoffs(), "payoffs")
print(env.is_over())

env.reset()
print(env.is_over())


# ret = env.step(random_agent.step(state), True)
# print(ret)


# Note: The trajectories are 3-dimension list. The first dimension is for different players.
# The second dimension is for different transitions. The third dimension is for the contents of each transiton

# [
#     [
#         {
#             "legal_actions": OrderedDict([(0, None), (1, None), (2, None)]),
#             "obs": array(
#                 [
#                     0.0,
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                 ]
#             ),
#             "raw_obs": {
#                 "hand": "HQ",
#                 "public_card": None,
#                 "all_chips": [1, 2],
#                 "my_chips": 1,
#                 "legal_actions": ["call", "raise", "fold"],
#                 "current_player": 0,
#             },
#             "raw_legal_actions": ["call", "raise", "fold"],
#             "action_record": [(0, "call"), (1, "fold")],
#         },
#         0,
#         {
#             "legal_actions": OrderedDict([(1, None), (2, None), (3, None)]),
#             "obs": array(
#                 [
#                     0.0,
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                 ]
#             ),
#             "raw_obs": {
#                 "hand": "HQ",
#                 "public_card": None,
#                 "all_chips": [2, 2],
#                 "my_chips": 2,
#                 "legal_actions": ["raise", "fold", "check"],
#                 "current_player": 0,
#             },
#             "raw_legal_actions": ["raise", "fold", "check"],
#             "action_record": [(0, "call"), (1, "fold")],
#         },
#     ],
#     [
#         {
#             "legal_actions": OrderedDict([(1, None), (2, None), (3, None)]),
#             "obs": array(
#                 [
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                 ]
#             ),
#             "raw_obs": {
#                 "hand": "HJ",
#                 "public_card": None,
#                 "all_chips": [2, 2],
#                 "my_chips": 2,
#                 "legal_actions": ["raise", "fold", "check"],
#                 "current_player": 1,
#             },
#             "raw_legal_actions": ["raise", "fold", "check"],
#             "action_record": [(0, "call"), (1, "fold")],
#         },
#         2,
#         {
#             "legal_actions": OrderedDict([(1, None), (2, None), (3, None)]),
#             "obs": array(
#                 [
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                     0.0,
#                 ]
#             ),
#             "raw_obs": {
#                 "hand": "HJ",
#                 "public_card": None,
#                 "all_chips": [2, 2],
#                 "my_chips": 2,
#                 "legal_actions": ["raise", "fold", "check"],
#                 "current_player": 0,
#             },
#             "raw_legal_actions": ["raise", "fold", "check"],
#             "action_record": [(0, "call"), (1, "fold")],
#         },
#     ],
# ]
