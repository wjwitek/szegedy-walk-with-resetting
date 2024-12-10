import numpy as np
from numpy.linalg import matrix_power

# The markov chain has to be row stochastic
class SzegedyRandomWalk:
    def __init__(self, markov_chain):
        self.markov_chain = markov_chain
        self.n = len(markov_chain)
        self.base = self.__construct_base()
        self.alphas = self._calculate_alphas()
        self.betas = self._calculate_betas()
        self.A, self.B = self.__construct_A_B()
        self.ref1, self.ref2 = self._construct_refs()
        self.operator = self._construct_evolution_operator()

    def __construct_base(self):
        base = []
        for i in range(self.n):
            base_i = np.zeros(self.n, dtype=np.float128)
            base_i[i] = 1
            base.append(base_i)
        return np.asarray(base, dtype=np.float128)

    def _calculate_alphas(self):
        alphas = []
        for i in range(self.n):
            alpha_i = np.zeros(self.n * self.n, dtype=np.float128)
            for j in range(self.n):
                alpha_i += np.sqrt(self.markov_chain[i, j], dtype=np.float128) * np.kron(self.base[i], self.base[j])
            alphas.append(alpha_i)
        return alphas

    def _calculate_betas(self):
        betas = []
        for i in range(self.n):
            beta_i = np.zeros(self.n * self.n, dtype=np.float128)
            for j in range(self.n):
                beta_i += np.sqrt(self.markov_chain[i, j], dtype=np.float128) * np.kron(self.base[j], self.base[i])
            betas.append(beta_i)
        return betas

    def __construct_A_B(self):
        return np.column_stack(tuple(self.alphas)), np.column_stack(tuple(self.betas))

    def _construct_refs(self):
        ref1 = 2 * self.A @ np.conjugate(self.A, dtype=np.float128).T - np.identity(self.n * self.n, dtype=np.float128)
        ref2 = 2 * self.B @ np.conjugate(self.B, dtype=np.float128).T - np.identity(self.n * self.n, dtype=np.float128)
        return ref1, ref2

    def _construct_evolution_operator(self):
        return self.ref2 @ self.ref1

    def state_at(self, time, initial_state, ):
        return matrix_power(self.operator, time) @ initial_state

    def original_state_at(self, time, initial_state):
        state_at = np.power(self.state_at(time, initial_state), 2)
        return state_at.reshape((self.n, self.n)).sum(axis=1)
