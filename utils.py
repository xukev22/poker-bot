def process_leduc_state_v1(raw_imperfect_state, pid):
    """we are given a lot of information in the leduc game state for a GIVEN player (note this is imperfect info)
    but we want some minimalist, consistent version that can be used for 'uniqueness' as well as for our conveience,
    this will help us to distinguish between what we care about and define some formal global definition

    by defining this, we also can abstract and encapsulate what pieces of state we care about allowing us to make variants of MC training

    in v1, we will do only the bare minimums: hand, public_card, opp_chips, my_chips
    note we'll disregard betting history/position/etc. in this simpliefied version

    an example of raw_imperfect_state structure:
    {'legal_actions': OrderedDict([(0, None), (1, None), (2, None)]), 'obs': array([1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0.]), 'raw_obs': {'hand': 'HJ', 'public_card': None, 'all_chips': [1, 2], 'my_chips': 2, 'legal_actions': ['call', 'raise', 'fold'], 'current_player': 0}, 'raw_legal_actions': ['call', 'raise', 'fold'], 'action_record': []}
    """
    raw_obs = raw_imperfect_state["raw_obs"]
    hand = raw_obs["hand"]
    public_card = raw_obs["public_card"]
    my_chips = raw_obs["all_chips"][pid]
    opp_chips = raw_obs["all_chips"][1 - pid]
    return (hand, public_card, my_chips, opp_chips)


def process_leduc_state_v2(raw_imperfect_state, pid):
    """we are given a lot of information in the leduc game state for a GIVEN player (note this is imperfect info)
    but we want some minimalist, consistent version that can be used for 'uniqueness' as well as for our conveience,
    this will help us to distinguish between what we care about and define some formal global definition

    by defining this, we also can abstract and encapsulate what pieces of state we care about allowing us to make variants of MC training

    in v1, we will do only the bare minimums: hand, public_card, opp_chips, my_chips
    note we'll disregard betting history/position/etc. in this simpliefied version

    an example of raw_imperfect_state structure:
    {'legal_actions': OrderedDict([(0, None), (1, None), (2, None)]), 'obs': array([1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0.]), 'raw_obs': {'hand': 'HJ', 'public_card': None, 'all_chips': [1, 2], 'my_chips': 2, 'legal_actions': ['call', 'raise', 'fold'], 'current_player': 0}, 'raw_legal_actions': ['call', 'raise', 'fold'], 'action_record': []}
    """
    raw_obs = raw_imperfect_state["raw_obs"]
    hand = raw_obs["hand"]
    public_card = raw_obs["public_card"]
    return (hand, public_card)


def process_leduc_state_v3(raw_imperfect_state, pid):
    """we are given a lot of information in the leduc game state for a GIVEN player (note this is imperfect info)
    but we want some minimalist, consistent version that can be used for 'uniqueness' as well as for our conveience,
    this will help us to distinguish between what we care about and define some formal global definition

    by defining this, we also can abstract and encapsulate what pieces of state we care about allowing us to make variants of MC training

    in v1, we will do only the bare minimums: hand, public_card, opp_chips, my_chips
    note we'll disregard betting history/position/etc. in this simpliefied version

    an example of raw_imperfect_state structure:
    {'legal_actions': OrderedDict([(0, None), (1, None), (2, None)]), 'obs': array([1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0.]), 'raw_obs': {'hand': 'HJ', 'public_card': None, 'all_chips': [1, 2], 'my_chips': 2, 'legal_actions': ['call', 'raise', 'fold'], 'current_player': 0}, 'raw_legal_actions': ['call', 'raise', 'fold'], 'action_record': []}
    """
    raw_obs = raw_imperfect_state["raw_obs"]
    hand = raw_obs["hand"]
    public_card = raw_obs["public_card"]
    my_chips = raw_obs["all_chips"][pid]
    opp_chips = raw_obs["all_chips"][1 - pid]
    # make immutable
    action_record = tuple(raw_imperfect_state["action_record"])
    return (hand, public_card, my_chips, opp_chips, action_record)
