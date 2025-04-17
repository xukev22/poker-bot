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

third issue: leduc is too simple

solution:
-> expanded to a variant of limit holdem

fourth issue: had to make a lot of changes
-> could not get termination (k_samples 50 vs. 5 insane reduction)
-> needed a new heuristic (new poker library used)

then:
-> exploring specific states (interesting findings/decisions)
-> playing around w/ heuristics (chip stacks, etc.) -> better results w/ pot size
-> normalize equities w/ sigmoid or something

next steps issue: does not capture reality of poker (imperfect info, variance)
planned solutions -> CFR
