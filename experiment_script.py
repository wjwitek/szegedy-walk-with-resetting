from utils import *

ht_50, prob_50 = graph_size_experiment_with_reset(get_cycle_graph, initial_state, 5, create_resetting_graph, resetting_rate=0.5)

with open("result.txt", "w") as f:
    f.write("Hitting time\n")
    f.write('\n'.join(list(map(str, ht_50))))
    f.write("\nProbability\n")
    f.write("\n".join(list(map(str, prob_50))))
