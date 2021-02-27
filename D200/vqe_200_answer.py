#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def variational_ansatz(params, wires):
    """The variational ansatz circuit.

    Fill in the details of your ansatz between the # QHACK # comment markers. Your
    ansatz should produce an n-qubit state of the form

        a_0 |10...0> + a_1 |01..0> + ... + a_{n-2} |00...10> + a_{n-1} |00...01>

    where {a_i} are real-valued coefficients.

    Args:
         params (np.array): The variational parameters.
         wires (qml.Wires): The device wires that this circuit will run on.
    """

    # QHACK #

    num_qubits = len(wires)
    params_id = 0

    if num_qubits == 1:
        qml.RY(params[0], wires=wires[0])

    else:
        for i in range(num_qubits-1):
            if i == 0:
                qml.RY(params[params_id], wires=wires[i])
                params_id += 1
                # X Gate
                qml.PauliX(wires=wires[i])

            else:
                if i == 1:
                    qml.CRY(params[params_id], wires=[wires[0], wires[1]])
                    params_id += 1
                else:
                    # MCRY Gate
                    control = [c for c in range(i)]
                    target = i

                    mult_cnot_qubit = len(control + [target])
                    unitary_matrix = np.zeros((2**mult_cnot_qubit, 2**mult_cnot_qubit))
                    np.fill_diagonal(unitary_matrix, val=1)
                    unitary_matrix[-1, :] *= 0
                    unitary_matrix[-1, -2] += 1
                    unitary_matrix[-2, :] *= 0
                    unitary_matrix[-2, -1] += 1

                    qml.RX(np.pi/2, wires=wires[target])
                    qml.RZ(params[params_id]/2, wires=wires[target])
                    qml.RX(-np.pi/2, wires=wires[target])
                    qml.QubitUnitary(unitary_matrix, wires=control + [target])  # MCCNOT
                    qml.RX(np.pi/2, wires=wires[target])
                    qml.RZ(-params[params_id]/2, wires=wires[target])
                    qml.RX(-np.pi/2, wires=wires[target])
                    qml.QubitUnitary(unitary_matrix, wires=control + [target])  # MCCNOT
                    params_id += 1
          
                # X Gate
                qml.PauliX(wires=wires[i])

        #CNOT
        if num_qubits > 3:
            #MCCNOT
            control = [c for c in range(num_qubits-1)]
            target = num_qubits-1
            mult_cnot_qubit = len(control + [target])
            unitary_matrix = np.zeros((2**mult_cnot_qubit, 2**mult_cnot_qubit))
            np.fill_diagonal(unitary_matrix, val=1)
            unitary_matrix[-1, :] *= 0
            unitary_matrix[-1, -2] += 1
            unitary_matrix[-2, :] *= 0
            unitary_matrix[-2, -1] += 1
            qml.QubitUnitary(unitary_matrix, wires=control + [target])
        else:
            if num_qubits == 2:
                qml.CNOT(wires=[wires[0], wires[1]])
            elif num_qubits == 3:
                qml.Toffoli(wires=[wires[0], wires[1], wires[2]])

        for i in range(num_qubits-1):
            qml.PauliX(wires=wires[i])

    # QHACK #


def run_vqe(H):
    """Runs the variational quantum eigensolver on the problem Hamiltonian using the
    variational ansatz specified above.

    Fill in the missing parts between the # QHACK # markers below to run the VQE.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The ground state energy of the Hamiltonian.
    """
    energy = 0

    # QHACK #
    min_energy = 100
    num_qubits = len(H.wires)
    if num_qubits == 1:
      num_params = 1
    else:
      num_params = num_qubits - 1

    # Initialize the quantum device
    dev = qml.device('default.qubit', wires=num_qubits)

    # Randomly choose initial parameters (how many do you need?)
    params = 0.01*np.random.randn(num_params)

    # Set up a cost function
    cost_fn = qml.ExpvalCost(variational_ansatz, H, dev)

    # Set up an optimizer
    opt = qml.optimize.AdamOptimizer(0.6)

    # Run the VQE by iterating over many steps of the optimizer
    max_iterations = 500
    conv_tol = 1e-8

    for n in range(max_iterations):
      params, prev_energy = opt.step_and_cost(cost_fn, params)
      energy = cost_fn(params)
      conv = np.abs(energy - prev_energy)

      if energy < min_energy:
        min_energy = energy

      if conv <= conv_tol:
          break

    energy = min_energy

    # QHACK #

    # Return the ground state energy
    return energy


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
    ground_state_energy = run_vqe(H)
    print(f"{ground_state_energy:.6f}")
