import numpy as np
from lib.szegedy import SzegedyRandomWalk
import matplotlib.pyplot as plt
from utils.graph_generators import *


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

def perform_walk(init_state, marked_matrix, marked_vertice, t=1000, stop_at_ht=True, full_history=False):
    current_state = init_state
    probabilities = []
    szegedy = SzegedyRandomWalk(marked_matrix)
    f_t = 0
    f_ts = []

    for i in range(t):
        f_t += np.power(current_state - init_state, 2).sum()
        f_ts.append(f_t / (i + 1))
        if full_history:
            probabilities.append(current_state)
        else:
            probabilities.append(extract_probability(current_state, marked_vertice))
        current_state = szegedy.operator @ current_state
        if stop_at_ht and f_t / (i + 1) >= 1 - 1 / len(init_state):
            break

    return probabilities, f_ts

def full_walk_experiment(graph_function, init_state_function, max_size, filename, max_t=100, marked_vertice=lambda x: x // 2, start=2, resetting_rate=0.3):
    with open(filename, 'w') as f:
        for i in range(start, max_size):
            unmarked, marked = graph_function(i, marked_vertice(i))
            if unmarked is None:
                continue
            initial_st = init_state_function(unmarked)
            f.write(f"{i}\n")
            probabilities, _ =  perform_walk(initial_st, marked, marked_vertice(i), t=max_t, stop_at_ht=False, full_history=True)
            f.write('\n'.join(','.join(map(str,sl)) for sl in probabilities))
            f.write('\n')


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

def graph_size_experiment(graph_name, init_state_function, max_size, max_t=1000, marked_vertice=lambda x: x // 2, start=2, resetting_rate=None, resetting_point=lambda x: 0):
    hitting_times = []
    probabilities = []

    for i in range(start, max_size):
        unmarked, marked = create_graph(graph_name, i, marked_vertice(i), resetting_rate, resetting_point(i))
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

def graph_size_experiment_with_reset(graph_name, init_state_function, max_size, reset_matrix_function, max_t=1000, mark_func=lambda x: x // 2, resetting_rate=None, resetting_point_func= lambda x: 0, start=2):
    hitting_times = []
    probabilities = []

    for i in range(start, max_size):
        unmarked, marked = create_graph(graph_name, i, mark_func(i), resetting_rate, resetting_point_func(i))
        if unmarked is None:
            continue
        initial_st = init_state_function(unmarked)
        reset_matrix = reset_matrix_function(i, resetting_rate, resetting_point_func(i))
        hitting_time, prob = get_quantum_hitting_time_with_reset(initial_st, marked, mark_func(i), reset_matrix, max_t)
        hitting_times.append(hitting_time)
        probabilities.append(prob)

    return hitting_times, probabilities
