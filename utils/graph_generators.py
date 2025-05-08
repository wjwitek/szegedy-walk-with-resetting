import numpy as np
import networkx as nx

graph_to_function_mapping = {"complete": lambda n: np.full((n, n), 1 / (n - 1)) - np.diag(np.full(n, 1 / (n - 1))),
                             "barbell": lambda n: nx.to_numpy_array(nx.barbell_graph((n - 2) // 2, 2 + (n - 2) % 2)),
                             "circular ladder": lambda n: nx.to_numpy_array(nx.circular_ladder_graph(n // 2)) if n % 2 == 0 else None,
                             "cycle": lambda n: nx.to_numpy_array(nx.cycle_graph(n)),
                             "star": lambda n: nx.to_numpy_array(nx.star_graph(n - 1)),
                             "balanced tree": lambda n: nx.to_numpy_array(nx.balanced_tree(2, find_balanced_tree_height(n)))[:n, :n]}

def create_graph(graph_name, n, marked_vertice=None, _resetting_rate=None, _resetting_point=0):
    if _resetting_rate is not None:
        return create_graph_with_embedded_resetting(graph_name, n, marked_vertice, _resetting_rate, _resetting_point)

    if marked_vertice is None:
        marked_vertice = n // 2

    # create unmarked graph
    unmarked = graph_to_function_mapping[graph_name](n)
    if unmarked is None:
        return None, None

    # fix edge weights
    for i in range(n):
        unmarked[i] /= np.sum(unmarked[i])

    # create marked graph
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def create_graph_with_embedded_resetting(graph_name, n, marked_vertice=None, resetting_rate=0.3, resetting_point=0):
    if marked_vertice is None:
        marked_vertice = n // 2

    # create unmarked graph
    unmarked = graph_to_function_mapping[graph_name](n)
    if unmarked is None:
        return None, None

    # add resetting
    for i in range(n):
        minus = unmarked[i] * resetting_rate / np.sum(unmarked[i])
        unmarked[i] = unmarked[i] / np.sum(unmarked[i]) - minus
    for i in range(n):
        unmarked[i, resetting_point] += resetting_rate

    # create marked graph
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def create_complete_graph(n, marked_vertice=None, resetting_rate=None):
    if marked_vertice is None:
        marked_vertice = n // 2
    unmarked = np.full((n, n), 1 / (n - 1)) - np.diag(np.full(n, 1 / (n - 1)))
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def create_complete_graph_with_embedded_resetting(n, marked_vertice=None, resetting_rate=0.3):
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

def create_resetting_graph(n, resetting_rate=0.3, resetting_point=0):
    graph = np.diag(np.full(n, 1 - resetting_rate))
    for i in range(1, n):
        graph[i, resetting_point] = resetting_rate
    graph[resetting_point, resetting_point] = 1
    return graph

def create_barbe_barbell_graph_with_resetting(n, marked_vertice=None, resetting_rate=0.3, resetting_point=None):
    if n < 6:
        return None, None
    if marked_vertice is None:
        marked_vertice = 0
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
        unmarked[i, n // 2 - 1] += resetting_rate

    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def get_circular_ladder_graph(n, marked_vertice=None, resetting_rate=0.3):
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

def get_cycle_graph(n, marked_vertice=None, resetting_rate=0.3, resetting_point=None):
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

def get_cycle_graph_with_embedded_resetting(n, marked_vertice=None, resetting_rate=0.3, resetting_point=None):
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

def create_star_graph(n, marked_vertice=None, resetting_rate=0.3):
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

def create_star_graph_with_embedded_resetting(n, marked_vertice=None, resetting_rate=0.3):
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

def create_barbe_barbell_graph(n, marked_vertice=None, resetting_rate=0.3, resetting_point=None):
    if n < 6:
        return None, None
    if marked_vertice is None:
        marked_vertice = 0
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

def find_balanced_tree_height(n):
    counter, h, nodes_in_layer = 1, 0, 1
    while counter < 1000:
        if counter >= n:
            break
        h += 1
        nodes_in_layer *= 2
        counter += nodes_in_layer
    return h

def create_balanced_tree(n, marked_vertice=None, resetting_rate=0.3):
    if marked_vertice is None:
        marked_vertice = n - 1
    # find correct height
    h = find_balanced_tree_height(n)
    G = nx.balanced_tree(2, h)
    unmarked = nx.to_numpy_array(G)[:n, :n]
    for i in range(n):
        unmarked[i] = unmarked[i] / np.sum(unmarked[i])
    # mark the edge
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked

def create_balanced_tree_with_resetting(n, marked_vertice=None, resetting_rate=0.3, resetting_point=0):
    if marked_vertice is None:
        marked_vertice = n - 1
    # find correct height
    h = find_balanced_tree_height(n)
    G = nx.balanced_tree(2, h)
    unmarked = nx.to_numpy_array(G)[:n, :n]
    if resetting_point is None:
        resetting_point = np.where(unmarked[marked_vertice] == 1)
    for i in range(n):
        minus = unmarked[i] * resetting_rate / np.sum(unmarked[i])
        unmarked[i] = unmarked[i] / np.sum(unmarked[i]) - minus
    for i in range(n):
        unmarked[i, resetting_point] += resetting_rate
    # mark the edge
    marked = np.copy(unmarked)
    marked_row = np.zeros(n)
    marked_row[marked_vertice] = 1
    marked[marked_vertice] = marked_row
    return unmarked, marked
