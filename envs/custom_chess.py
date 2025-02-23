
from pettingzoo.utils.wrappers import BaseWrapper


class CustomChess(BaseWrapper):

    def __init__(self,env):
        super().__init__(env)

    def step(self,action):

        action = int(action)
        ret = super(action)

        return ret