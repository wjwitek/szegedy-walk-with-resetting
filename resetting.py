from szegedy import SzegedyRandomWalk
import numpy as np

class AlternatingResetting(SzegedyRandomWalk):
    def __init__(self, markov_chain, resetting_chain, col_stochastic=False, resetting_frequency=2):
        if len(markov_chain) != len(resetting_chain):
            raise ValueError("Both base markov chain and the resetting chain must have the same order!")
        if resetting_frequency < 2:
            raise ValueError("Resetting frequency must be higher than 1!")
        super().__init__(markov_chain, col_stochastic)
        self.resetting_chain = resetting_chain
        self.resetting_alphas = self._calculate_alphas(resetting_chain)
        self.resetting_betas = self._calculate_betas(resetting_chain)
        self.pi_A_resetting, self.pi_B_resetting = self._construct_pi_A_B(self.resetting_alphas, self.resetting_betas)
        self.resetting_operator = self._construct_evolution_operator(self.pi_A_resetting, self.pi_B_resetting)
        self.resetting_frequency = resetting_frequency

    def state_at(self, time, initial_state, as_probability=False):
        result = self.operator @ initial_state
        for t in range(1, time):
            if t % self.resetting_frequency == 0:
                result = self.resetting_operator @ result
            else:
                result = self.operator @ result
        if as_probability:
            return np.square(result)
        else:
            return result
