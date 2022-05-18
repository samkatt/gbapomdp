"""Runs some gridworld functions a bunch of times"""
import time

from general_bayes_adaptive_pomdps.domains.gridworld import GridWorld


def main():
    d = GridWorld(7, True)
    n = 500000

    states = [d.state_space.sample() for _ in range(n)]
    actions = [d.action_space.sample_as_int() for _ in range(len(states))]

    t = time.time()

    for s, a in zip(states, actions):
        d.simulation_step(s, a)

    print(f"Took {time.time() - t} seconds")


if __name__ == "__main__":
    main()
