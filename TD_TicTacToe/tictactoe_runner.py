from tictactoe import tictactoe
from td_agent import td_agent

import numpy as np

if __name__ == '__main__':
    PATH = 'tictactoe_learned_vals/'
    xvals = np.load(PATH+'x_game_vals_fixed.npy')
    ovals = 1 - xvals

    game = tictactoe()
    state_arr = game.gen_state_arr()
    choose = str(input('Select piece to play as: input X or O:')).upper()
    assert choose == 'X' or choose == 'O'

    if choose == 'X':
        agent = td_agent(ovals, .1)
        game.human_play(state_arr, agent, 2)

    else:
        agent = td_agent(xvals, .1)
        game.human_play(state_arr, agent, 1)
