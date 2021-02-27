# Qhack 2021 Solutions
This repo contains my team (Entangled_Nets) submission for the QML Challenges in [QHack 2021](https://github.com/XanaduAI/QHack), a quantum machine learning hackathon. We achieved perfect score (2500).

The challenges are:
## A: Simple Circuits
1. A20: Measurement
> Calculate the probability of a rotated qubit is in the ground state.
2. A30: Expectation Values
> Evaluate an expectation value for a measurement of a rotated qubit.
3. A50: Entanglement
> Calculate a tensor-product observable for an entangled state. 

## B: Quantum Gradients
1. B100: Exploring Quantum Gradients
> Compute the gradient of the provided QNode (a quantum circuit on a particular device) using the parameter-shift rule.
2. B200: Higher-Order Derivatives
> Given a variational quantum circuit, compute the gradient and the Hessian of the circuit using the parameter-shift rule by hand (do not use PennyLane’s built-in gradient methods).
3. B500: Finding the Natural Gradient
> Calculate the Fubini-Study metric and using it to find the quantum natural gradient (QNG).

## C: Circuit Training
1. C100: Optimizing a Quantum Circuit
> Provided with a variational quantum circuit, find the minimum expectation value this circuit can produce by optimizing its parameters.
2. C200: QAOA
> Set up a QAOA circuit in PennyLane and use pre-optimized parameters to identify the maximum independent set of a graph with six nodes.
3. C500: Variational Quantum Classifier
> Design a variational quantum classifier that can classify unknown test data from the same distribution of the given data with an accuracy of more than 95%.
> Helpful paper: [Data re-uploading for a universal quantum classifier](https://quantum-journal.org/papers/q-2020-02-06-226/).

## D: VQE
1. D100: Optimization Orchestrator
> Implement the classical control flow and optimization portion of the VQE to find the ground state energy of a given Hamiltonian.
2. D200: Ansatz Artistry
> Design an ansatz for a class of Hamiltonians whose n-qubit eigenstates must have the form:
> ![equation](https://bit.ly/3bEtjmC).
3. D500: Moving On Up
> Implement a variational method that will find the ground state, as well as the first two excited states of the provided Hamiltonian.
> Helpful paper: [Variational Quantum Computation of Excited States](https://arxiv.org/abs/1805.08138).
