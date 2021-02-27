#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #

    def variational_ansatz(params, wires):
        """
        Args:
            params (np.ndarray): An array of floating-point numbers with size (n, 3),
                where n is the number of parameter sets required (this is determined by
                the problem Hamiltonian).
            wires (qml.Wires): The device wires this circuit will run on.
        """
        n_qubits = len(wires)
        n_rotations = len(params)

        if n_rotations > 1:
            n_layers = n_rotations // n_qubits
            n_extra_rots = n_rotations - n_layers * n_qubits

            # Alternating layers of unitary rotations on every qubit followed by a
            # ring cascade of CNOTs.
            for layer_idx in range(n_layers):
                layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
                qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
                qml.broadcast(qml.CNOT, wires, pattern="ring")

            # There may be "extra" parameter sets required for which it's not necessarily
            # to perform another full alternating cycle. Apply these to the qubits as needed.
            extra_params = params[-n_extra_rots:, :]
            extra_wires = wires[: n_qubits - 1 - n_extra_rots : -1]
            qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
        else:
            # For 1-qubit case, just a single rotation to the qubit
            qml.Rot(*params[0], wires=wires[0])


    num_qubits = len(H.wires)

    # Initialize the quantum device
    dev = qml.device('default.qubit', wires=num_qubits)

    # Randomly choose initial parameters
    #params = 0.01*np.random.randn(num_params)
    num_param_sets = (2 ** num_qubits) - 1
    params_0 = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))
    params_1 = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))
    params_2 = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))


    # FIRST ITER: FIND THE GS ENERGY
    # Set up a cost function for ground state
    cost_fn_0 = qml.ExpvalCost(variational_ansatz, H, dev)
    # Set up an optimizer
    opt = qml.optimize.AdamOptimizer(0.6)
    # Run the VQE by iterating over many steps of the optimizer
    max_iterations = 500
    conv_tol = 1e-6
    min_loss = 100
    for n in range(max_iterations):
      params_0, prev_loss = opt.step_and_cost(cost_fn_0, params_0)
      loss = cost_fn_0(params_0)
      conv = np.abs(loss - prev_loss)

      if loss < min_loss:
        best_params_0 = params_0
        min_loss = loss

      if conv <= conv_tol:
          break

    energies[0] = cost_fn_0(best_params_0)


    # SECOND ITER: FIND THE 1ST ES ENERGY
    # Set up a cost function for ground state
    dev_swap_01 = qml.device('default.qubit', wires=2*num_qubits + 1)
    @qml.qnode(dev_swap_01)
    def swap_test_01(params, wires):

        Q = len(wires)
        # load the two inputs into two different registers
        variational_ansatz(best_params_0, wires=[q for q in range(int(Q/2))])
        variational_ansatz(params, wires=[q + int(Q/2) for q in range(int(Q/2))])

        # perform the SWAP test
        qml.Hadamard(wires=Q-1)
        for k in range(int(Q/2)):
            qml.CSWAP(wires=[Q-1, k, int(Q/2) + k])
        qml.Hadamard(wires=Q-1)

        return qml.expval(qml.PauliZ(Q-1))

    cost_fn_1 = qml.ExpvalCost(variational_ansatz, H, dev)
    def total_cost_1(params):
      return cost_fn_1(params) + abs(energies[0])*swap_test_01(params, wires=[c for c in range(2*num_qubits + 1)])
    # Set up an optimizer
    opt = qml.optimize.AdamOptimizer(0.6)
    # Run the VQE by iterating over many steps of the optimizer
    max_iterations = 500
    conv_tol = 1e-6
    min_loss = 100
    for n in range(max_iterations):
      params_1, prev_loss = opt.step_and_cost(total_cost_1, params_1)
      loss = total_cost_1(params_1)
      conv = np.abs(loss - prev_loss)

      if loss < min_loss:
        best_params_1 = params_1
        min_loss = loss

      if conv <= conv_tol:
          break

    energies[1] = cost_fn_1(best_params_1)
    
    
    # THIRD ITER: FIND THE 2ND ES ENERGY
    # Set up a cost function for ground state
    dev_swap_02 = qml.device('default.qubit', wires=2*num_qubits + 1)
    @qml.qnode(dev_swap_02)
    def swap_test_02(params, wires):

        Q = len(wires)
        # load the two inputs into two different registers
        variational_ansatz(best_params_0, wires=[q for q in range(int(Q/2))])
        variational_ansatz(params, wires=[q + int(Q/2) for q in range(int(Q/2))])

        # perform the SWAP test
        qml.Hadamard(wires=Q-1)
        for k in range(int(Q/2)):
            qml.CSWAP(wires=[Q-1, k, int(Q/2) + k])
        qml.Hadamard(wires=Q-1)

        return qml.expval(qml.PauliZ(Q-1))

    dev_swap_12 = qml.device('default.qubit', wires=2*num_qubits + 1)
    @qml.qnode(dev_swap_12)
    def swap_test_12(params, wires):

        Q = len(wires)
        # load the two inputs into two different registers
        variational_ansatz(best_params_1, wires=[q for q in range(int(Q/2))])
        variational_ansatz(params, wires=[q + int(Q/2) for q in range(int(Q/2))])

        # perform the SWAP test
        qml.Hadamard(wires=Q-1)
        for k in range(int(Q/2)):
            qml.CSWAP(wires=[Q-1, k, int(Q/2) + k])
        qml.Hadamard(wires=Q-1)

        return qml.expval(qml.PauliZ(Q-1))
    
    cost_fn_2 = qml.ExpvalCost(variational_ansatz, H, dev)
    wires_cost2 = [c for c in range(2*num_qubits + 1)]
    def total_cost_2(params):
      return cost_fn_2(params) + abs(energies[0])*swap_test_02(params, wires=wires_cost2) + abs(energies[1])*swap_test_12(params, wires=wires_cost2)
    # Set up an optimizer
    opt = qml.optimize.AdamOptimizer(0.6)
    # Run the VQE by iterating over many steps of the optimizer
    max_iterations = 500
    conv_tol = 1e-6
    min_loss = 100
    for n in range(max_iterations):
      params_2, prev_loss = opt.step_and_cost(total_cost_2, params_2)
      loss = total_cost_2(params_2)
      conv = np.abs(loss - prev_loss)

      if loss < min_loss:
        best_params_2 = params_2
        min_loss = loss

      if conv <= conv_tol:
          break

    energies[2] = cost_fn_2(best_params_2)

    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
