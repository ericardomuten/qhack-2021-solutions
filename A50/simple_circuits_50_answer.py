#! /usr/bin/python3
import pennylane as qml
from pennylane import numpy as np
import sys


def simple_circuits_50(angle):
    """The code you write for this challenge should be completely contained within this function
        between the # QHACK # comment markers.

    In this function:
        * Create the standard Bell State
        * Rotate the first qubit around the y-axis by angle
        * Measure the expectation value of the tensor observable `qml.PauliZ(0) @ qml.PauliZ(1)`

    Args:
        angle (float): how much to rotate a state around the y-axis

    Returns:
        float: the expectation value of the tensor observable
    """

    expectation_value = 0.0

    # QHACK #

    # Step 1 : initialize a device
    num_wires = 2
    dev = qml.device('default.qubit', wires=num_wires)

    # Step 2 : Create a quantum circuit and qnode
    @qml.qnode(dev)
    def my_quantum_function(angle):
        qml.Hadamard(wires=0) # a single-wire parameterized gate qml.CNOT(wires=[0, 1]) # a two-wire gate
        qml.CNOT(wires=[0, 1])
        qml.RX(angle, wires=0)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    
   
    # Step 3 : Run the qnode
    expectation_value =float(my_quantum_function(angle))

  
    # QHACK #
    return expectation_value


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    #print("mai")
    # Load and process input
    angle_str = sys.stdin.read()
    #print("mai")
    angle = float(angle_str)
    #print("mai")
    ans = simple_circuits_50(angle)

    if isinstance(ans, np.tensor):
        ans = ans.item()

    if not isinstance(ans, float):
        raise TypeError(
            "the simple_circuits_50 function needs to return either a float or PennyLane tensor."
        )

    print(ans)
