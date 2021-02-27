#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    
    
    # gradient 

    # hesian
    s=np.pi/2
    cl=0
    for i in range(5):
        for j in range(i,5):
            if j>i:
                shifted = weights.copy()
                shifted[i] += s
                shifted[j] += s
                a_1 = circuit(shifted) # ++

                shifted[i] -= 2*s
                a_2 = circuit(shifted) # - +

                shifted[j] -= 2*s
                a_4 = circuit(shifted) # - -

                shifted[i] += 2*s
                a_3 = circuit(shifted) # - -
                
                hessian[i][j] =  (a_1-a_2-a_3+a_4)/((2*np.sin(s))**2)
                hessian[j][i] = hessian[i][j]
                
            if j==i: 
                if cl==0:
                    cl=cl+1
                    shifted = weights.copy()
                    clasic=circuit(shifted)
                shifted = weights.copy()
                shifted[i] += s
                forward=circuit(shifted)
                
                shifted[i] -= 2*s
                backward=circuit(shifted)
                
                hessian[i][j] =  (forward-2*clasic+backward)/(2)
                gradient[i] = 0.5 * (forward - backward)/(np.sin(s))
            
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    #print("ok")
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
