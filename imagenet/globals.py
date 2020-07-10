def initialize():
    global idx
    idx = 0

def initialize_pruning_params():
    global parameters_to_prune
    parameters_to_prune = []
    global pruned_weights
    pruned_weights = []
    global zero_weights
    zero_weights = 0
    global total_weights
    total_weights = 0