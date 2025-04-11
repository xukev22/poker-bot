import rlcard

env = rlcard.make("leduc-holdem")

env.reset()

print(env.get_perfect_information())
