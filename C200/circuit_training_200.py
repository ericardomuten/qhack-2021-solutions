#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.
    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.
    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)
    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []

# QHACK #
    #device
    dev = dev = qml.device("default.qubit", wires=NODES, analytic=True, shots=1)
    #hamiltonian
    cost_h, mixer_h = qml.qaoa.max_independent_set(graph, constrained=True)
    wires=[0,1,2,3,4,5]
    
    
    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(alpha, cost_h)
        qml.qaoa.mixer_layer(gamma, mixer_h)
        
    def comp_basis_measurement(wires):
        n_wires = len(wires)
        return qml.Hermitian(np.diag(range(2 ** n_wires)), wires=wires)
    
    
    def circuit(params,**kwargs):

        #for w in range(6):
            #qml.Hadamard(wires=w)

        qml.layer(qaoa_layer,10, params[0], params[1])
        
    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=wires)
    
    cost_function = qml.ExpvalCost(circuit, cost_h, dev)
    optimizer = qml.GradientDescentOptimizer(stepsize = 0.01)
    steps = 0
    for i in range(steps):
        params = optimizer.step(cost_function, params)
        
    prob = probability_circuit(params[1], params[0])
    
    a=np.argmax(prob)
    s=[0,0,0,0,0,0]
    i=0
    if a==0:
        max_ind_set.append(0)
    while a!=0:
        s[5-i]=int(a)%2
        a=int(a/2)
        i+=1
    for i in range(len(s)):
        if s[i]==1:
            max_ind_set.append(i)
    # QHACK #




    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)