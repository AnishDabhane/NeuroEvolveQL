rewards_Arr = ['decide this']

class env:
    def __init__(self, state):
        self.current_state = state

    def step(self, action):
        next_state = self.current_state.copy()
        reward = 0
        done = False
        for i in range(len(self.current_state)):
            pass
##################################################################################################
        #should be changed
        #next_state[i] -= 10 * rewards_Arr[action][i]
        #reward += 10 * rewards_Arr[action][i]
        # self.current_state = next_state     imp. check use of this statement
##################################################################################################

        return next_state, reward, done

