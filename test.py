import gym
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, next_mark
from algorithm import MonteCarloTreeSearch


def run():
    s = env.reset()
    while env.done == False:
        if s[1] == 'O':
            action = t.search()
            s_, r, done, info = env.step(action)
        else:
            print(env.available_actions())
            action = int(input(''))
            s_, r, done, info = env.step(action)
            t.to_child(s_)
        env.render()
    s = s_
    t.reset_root()


if __name__ == "__main__":
    env = TicTacToeEnv()
    t = MonteCarloTreeSearch(30000, player='O')
    run()
