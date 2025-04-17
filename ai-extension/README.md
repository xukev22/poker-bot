first issue: RLcard does not extend well to search
-> hacky and had to mess w/ internal implementation details and still couldnt get it to work

solution:
-> switched libraries to openspiel
-> different control mechanism allowed me to control chance nodes

second issue: expectiminimax was returning very similar results:

-> see first graph
-> only 0, -1, 1
-> only call was used

solution:
-> played around w/ heuristics
-> played around w/ depth
-> played around w/ starting hands
-> exploring specific states (interesting findings/decisions)
-> extending to bigger variants of game (limit holdem)

third issue: does not capture reality of poker (imperfect info, variance)
