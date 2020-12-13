import os
import sys
sys.path.insert(0, os.path.abspath('./rlcard'))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import rlcard
from DQNAgent import DQNAgent
from rlcard.agents import LimitholdemHumanAgent as HumanAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from rlcard.utils.utils import print_card

set_global_seed(0)

env = rlcard.make('limit-holdem', config={'record_action': True})
human_agent = HumanAgent(env.action_num)

dqn_agent = DQNAgent(env.action_num,
                     env.state_shape[0],
                     hidden_neurons=[1024, 512, 1024, 512])

dqn_agent.load(os.path.join('first-limit-dqn-model', 'model.pth'))

env.set_agents([human_agent, dqn_agent])


def play():
    while (True):
        print(">> Start a new game")

        trajectories, payoffs = env.run(is_training=False)
        # If the human does not take the final action, we need to
        # print other players action
        if len(trajectories[0]) != 0:
            final_state = trajectories[0][-1][-2]
            action_record = final_state['action_record']
            _action_list = []
            for i in range(1, len(action_record)+1):
                """
                if action_record[-i][0] == state['current_player']:
                    break
                """
                _action_list.insert(0, action_record[-i])
            for pair in _action_list:
                print('>> Player', pair[0], 'chooses', pair[1])

        # Let's take a look at what the agent card is
        print('=============     DQN    ============')
        print_card(env.get_perfect_information()['hand_cards'][1])

        print('===============     Result     ===============')
        if payoffs[0] > 0:
            print('You win {} chips!'.format(payoffs[0]))
        elif payoffs[0] == 0:
            print('It is a tie.')
        else:
            print('You lose {} chips!'.format(-payoffs[0]))
        print('')

        input("Press any key to continue...")


play()
