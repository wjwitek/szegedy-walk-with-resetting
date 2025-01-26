import numpy as np
from szegedy import SzegedyRandomWalk
import matplotlib.pyplot as plt
import networkx as nx


def initial_state(unmarked_matrix):
    n = len(unmarked_matrix)
    basis = np.identity(n)
    initial_vector = np.zeros(n * n)
    for i in range(n):
        for j in range(n):
            initial_vector += np.sqrt(unmarked_matrix[i][j]) * np.kron(basis[i], basis[j])
    return initial_vector / np.sqrt(n)

def extract_probability(vector, marked_vertice):
    n = int(np.sqrt(len(vector)))
    return np.power(vector, 2).reshape(n, n).sum(axis=1)[marked_vertice]

def create_complete_graph(n, marked_vertice=None):
    if marked_vertice is None:
        marked_vertice = n // 2
    unmarked = np.full((n, n), 1 / (n - 1)) - np.diag(np.full(n, 1 / (n - 1)))
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def create_complete_graph_with_embedded_resetting(n, resetting_rate=0.3, marked_vertice=None):
    if marked_vertice is None:
        marked_vertice = n // 2
    move_prob = (1 - resetting_rate) / (n - 1)
    unmarked = np.full((n, n), move_prob) - np.diag(np.full(n, move_prob))
    for i in range(n):
        unmarked[i, 0] = resetting_rate
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def create_resetting_graph(n, resetting_rate=0.3):
    graph = np.diag(np.full(n, 1 - resetting_rate))
    for i in range(1, n):
        graph[i, 0] = resetting_rate
    graph[0, 0] = 1
    return graph

def perform_walk(init_state, marked_matrix, marked_vertice, t=1000):
    current_state = init_state
    probabilities = []
    szegedy = SzegedyRandomWalk(marked_matrix)
    f_t = 0
    f_ts = []

    for i in range(t):
        f_t += np.power(current_state - init_state, 2).sum()
        f_ts.append(f_t / (i + 1))
        probabilities.append(extract_probability(current_state, marked_vertice))
        current_state = szegedy.operator @ current_state
        if f_t / (i + 1) >= 1 - 1 / len(init_state):
            break

    plt.plot(list(range(len(probabilities))), probabilities)
    plt.show()

    return probabilities, f_ts

def get_quantum_hitting_time(init_state, marked_matrix, marked_vertice, max_t=1000):
    current_state = init_state
    szegedy = SzegedyRandomWalk(marked_matrix)
    f_t = 0
    f_ts = []
    t = 0

    for i in range(max_t):
        f_t += np.power(current_state - init_state, 2).sum()
        f_ts.append(f_t / (i + 1))
        t = i + 1
        if f_t / (i + 1) >= 1 - 1 / len(init_state):
            break
        current_state = szegedy.operator @ current_state

    return t, extract_probability(current_state, marked_vertice)

def graph_size_experiment(graph_function, init_state_function, max_size, max_t=1000, marked_vertice=lambda x: x // 2, start=2, resetting_rate=0.3):
    hitting_times = []
    probabilities = []

    for i in range(start, max_size):
        unmarked, marked = graph_function(i, resetting_rate)
        if unmarked is None:
            continue
        initial_st = init_state_function(unmarked)
        hitting_time, prob = get_quantum_hitting_time(initial_st, marked, marked_vertice(i), max_t)
        hitting_times.append(hitting_time)
        probabilities.append(prob)
        if i % 5 == 0:
            print(f"Completed step {i}.")

    return hitting_times, probabilities

def perform_walk_with_reset(init_state, marked_matrix, marked_vertice, reset_matrix, t=1000):
    current_state = init_state
    probabilities = []
    szegedy = SzegedyRandomWalk(marked_matrix)
    reset_szegedy = SzegedyRandomWalk(reset_matrix)
    f_t = 0
    f_ts = []

    for i in range(t):
        f_t += np.power(current_state - init_state, 2).sum()
        f_ts.append(f_t / (i + 1))
        probabilities.append(extract_probability(current_state, marked_vertice))
        current_state = szegedy.operator @ current_state
        current_state = reset_szegedy.operator @ current_state
        if f_t / (i + 1) >= 1 - 1 / len(init_state):
            break

    plt.plot(list(range(len(probabilities))), probabilities)
    plt.show()

    return probabilities, f_ts

def get_quantum_hitting_time_with_reset(init_state, marked_matrix, marked_vertice, reset_matrix, max_t=1000):
    current_state = init_state
    szegedy = SzegedyRandomWalk(marked_matrix)
    reset_szegedy = SzegedyRandomWalk(reset_matrix)
    f_t = 0
    f_ts = []
    t = 0

    for i in range(max_t):
        f_t += np.power(current_state - init_state, 2).sum()
        f_ts.append(f_t / (i + 1))
        t = i + 1
        if f_t / (i + 1) >= 1 - 1 / len(init_state):
            break
        current_state = szegedy.operator @ current_state
        current_state = reset_szegedy.operator @ current_state

    return t, extract_probability(current_state, marked_vertice)

def graph_size_experiment_with_reset(graph_function, init_state_function, max_size, reset_matrix_function, max_t=1000, mark_func=lambda x: x // 2, resetting_rate=0.3):
    hitting_times = []
    probabilities = []

    for i in range(2, max_size):
        unmarked, marked = graph_function(i)
        if unmarked is None:
            continue
        initial_st = init_state_function(unmarked)
        reset_matrix = reset_matrix_function(i, resetting_rate)
        hitting_time, prob = get_quantum_hitting_time_with_reset(initial_st, marked, mark_func(i), reset_matrix, max_t)
        hitting_times.append(hitting_time)
        probabilities.append(prob)

    return hitting_times, probabilities

def get_cycle_graph(n, marked_vertice=None):
    if marked_vertice is None:
        marked_vertice = n // 2
    G = nx.cycle_graph(n)
    unmarked = nx.to_numpy_array(G)
    unmarked[unmarked > 0] = 0.5
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def get_cycle_graph_with_embedded_resetting(n, resetting_rate=0.3, marked_vertice=None):
    if marked_vertice is None:
        marked_vertice = n // 2
    G = nx.cycle_graph(n)
    unmarked = nx.to_numpy_array(G)
    unmarked[unmarked > 0] = (1 - resetting_rate) / 2
    for i in range(n):
        unmarked[i, 0] += resetting_rate
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def create_star_graph(n, marked_vertice=None):
    if marked_vertice is None:
        marked_vertice = n // 2
    G = nx.star_graph(n - 1)
    unmarked = nx.to_numpy_array(G)
    unmarked[unmarked > 0] = 1.0
    unmarked[0] = unmarked[0] / (n - 1)
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def create_star_graph_with_embedded_resetting(n, resetting_rate=0.3, marked_vertice=None):
    if marked_vertice is None:
        marked_vertice = n // 2
    G = nx.star_graph(n - 1)
    unmarked = nx.to_numpy_array(G)
    unmarked[unmarked > 0] = 1.0 - resetting_rate
    unmarked[0] = (1 - resetting_rate) / (n - 1)
    unmarked[0, 0] = 0
    for i in range(n):
        unmarked[i, 1] += resetting_rate
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def create_barbe_barbell_graph(n, marked_vertice=None):
    if n < 6:
        return None, None
    if marked_vertice is None:
        marked_vertice = n // 2
    G = nx.barbell_graph((n - 2) // 2, 2 + (n - 2) % 2)
    unmarked = nx.to_numpy_array(G)
    # inside barbell
    unmarked[unmarked > 0] = 1 / ((n - 2) // 2 - 1)
    # on the edge
    t_1, t_2 = (n - 2) // 2 - 1, (n - 2) // 2 + 2 + (n - 2) % 2
    unmarked[t_1][unmarked[t_1] > 0] = 1 / ((n - 2) // 2)
    unmarked[t_2][unmarked[t_2] > 0] = 1 / ((n - 2) // 2)
    # path between
    unmarked[t_1+1:t_2][unmarked[t_1+1:t_2] > 0] = 1/2

    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

#%% md
### Barbell graph, embedded resetting
#%%
def create_barbe_barbell_graph_with_resetting(n, resetting_rate=0.3, marked_vertice=None):
    if n < 6:
        return None, None
    if marked_vertice is None:
        marked_vertice = n // 2
    G = nx.barbell_graph((n - 2) // 2, 2 + (n - 2) % 2)
    unmarked = nx.to_numpy_array(G)
    # inside barbell
    unmarked[unmarked > 0] = (1 - resetting_rate) / ((n - 2) // 2 - 1)
    # on the edge
    t_1, t_2 = (n - 2) // 2 - 1, (n - 2) // 2 + 2 + (n - 2) % 2
    unmarked[t_1][unmarked[t_1] > 0] = (1 - resetting_rate) / ((n - 2) // 2)
    unmarked[t_2][unmarked[t_2] > 0] = (1 - resetting_rate) / ((n - 2) // 2)
    # path between
    unmarked[t_1 + 1:t_2][unmarked[t_1 + 1:t_2] > 0] = (1 - resetting_rate) / 2

    for i in range(n):
        unmarked[i, 0] += resetting_rate

    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def get_circular_ladder_graph(n, marked_vertice=None):
    if n % 2 == 1:
        return None, None
    if marked_vertice is None:
        marked_vertice = n // 4
    G = nx.circular_ladder_graph(n // 2)
    unmarked = nx.to_numpy_array(G)
    unmarked[unmarked > 0] = 1 / 3
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def get_circular_ladder_graph_with_reset(n, marked_vertice=None, resetting_rate=0.3):
    if n % 2 == 1:
        return None, None
    if marked_vertice is None:
        marked_vertice = n // 4
    G = nx.circular_ladder_graph(n // 2)
    unmarked = nx.to_numpy_array(G)
    unmarked[unmarked > 0] = (1 - resetting_rate) / 3
    for i in range(n):
        unmarked[i, 0] += resetting_rate
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked
