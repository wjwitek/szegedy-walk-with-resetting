import numpy as np
from numpy.linalg import matrix_power
import networkx as nx

# This was based on the book, but needs to be rewritten as it got messy.
class SzegedyRandomWalkOld:
    def __init__(self, markov_chain, col_stochastic=False):
        self.markov_chain = markov_chain
        self.col_stochastic = col_stochastic
        self.n = len(markov_chain)
        self.base = self.__construct_base()
        self.alphas = self._calculate_alphas()
        self.betas = self._calculate_betas()
        self.A, self.B = self.__construct_A_B()
        self.pi_A, self.pi_B = self._construct_pi_A_B()
        self.operator = self._construct_evolution_operator()

    def classic_initial_state(self):
        state = np.zeros(self.n * self.n)
        for alpha in self.alphas:
            state += alpha
        return state / np.sqrt(self.n)

    def state_at(self, time, initial_state, as_probability=False):
        result = matrix_power(self.operator, time) @ initial_state
        if as_probability:
            return np.square(np.abs(result))
        else:
            return result

    def state_history(self, time, initial_state, as_probability=False):
        states = []
        for i in range(time):
            states.append(self.state_at(i, initial_state, as_probability=as_probability))
        return np.asarray(states)

    def original_state_at(self, time, initial_state):
        state_at = self.state_at(time, initial_state, True)
        return state_at.reshape((self.n, self.n)).sum(axis=1)

    def original_state_history(self, time, initial_state):
        states = []
        for i in range(time):
            states.append(self.original_state_at(i, initial_state))
        return states

    def __construct_base(self):
        base = []
        for i in range(self.n):
            base_i = np.zeros(self.n)
            base_i[i] = 1
            base.append(base_i)
        return np.asarray(base)

    def _calculate_alphas(self, chain=None):
        if chain is None:
            chain = self.markov_chain
        alphas = []
        for i in range(self.n):
            alpha_i = np.zeros(self.n * self.n)
            for j in range(self.n):
                if self.col_stochastic:
                    alpha_i += np.sqrt(chain[i, j]) * np.kron(self.base[i], self.base[j])
                else:
                    alpha_i += np.sqrt(chain[j, i]) * np.kron(self.base[i], self.base[j])
            alphas.append(alpha_i)
        return alphas

    def _calculate_betas(self, chain=None):
        if chain is None:
            chain = self.markov_chain
        betas = []
        for i in range(self.n):
            beta_i = np.zeros(self.n * self.n)
            for j in range(self.n):
                if self.col_stochastic:
                    beta_i += np.sqrt(chain[i, j]) * np.kron(self.base[j], self.base[i])
                else:
                    beta_i += np.sqrt(chain[j, i]) * np.kron(self.base[j], self.base[i])
            betas.append(beta_i)
        return betas

    def __construct_A_B(self, alphas=None, betas=None):
        if alphas is None or betas is None:
            alphas = self.alphas
            betas = self.betas
        A = np.zeros((self.n * self.n, self.n))
        B = np.zeros((self.n * self.n, self.n))
        for i in range(self.n):
            A += np.tensordot(alphas[i], self.base[i].T, axes=0)
            B += np.tensordot(betas[i], self.base[i].T, axes=0)
        return A, B

    def _construct_pi_A_B(self, alphas=None, betas=None):
        if alphas is None or betas is None:
            alphas = self.alphas
            betas = self.betas
        A = np.zeros((self.n * self.n, self.n * self.n))
        B = np.zeros((self.n * self.n, self.n * self.n))

        for i in range(self.n):
            A += np.tensordot(alphas[i], np.conjugate(alphas[i]).T, axes=0)
            B += np.tensordot(betas[i], np.conjugate(betas[i]).T, axes=0)

        return A, B

    def _construct_evolution_operator(self, pi_A=None, pi_B=None):
        if pi_A is None or pi_B is None:
            pi_A = self.pi_A
            pi_B = self.pi_B
        r_a = 2 * pi_A - np.identity(self.n * self.n)
        r_b = 2 * pi_B - np.identity(self.n * self.n)

        return np.matmul(r_a, r_b)


def get_graph(matrix):
    graph = nx.DiGraph()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                continue
            graph.add_weighted_edges_from([(i + 1, j + 1, matrix[i, j])])
    pos = nx.spring_layout(graph)
    nx.draw(graph, with_labels=True, pos=pos)
    nx.draw_networkx_edge_labels(graph, pos=pos)
