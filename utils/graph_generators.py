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

