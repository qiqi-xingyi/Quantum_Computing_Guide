# Quantum Computing Guide
![Image text](https://github.com/qiqi-xingyi/Quantum_Computing_Guide/blob/main/img/0.png)
A quantum computing tutorial for engineering students starting from scratch.

This tutorial will start from the basics of mathematics and quantum mechanics, and in very simple and clear language, it will help you understand the fundamental concepts of quantum computing and guide you on how to build quantum circuits and perform numerical simulations.

This guide requires you to have some basic knowledge of linear algebra (e.g., knowing what a matrix is) and some familiarity with Python syntax (this is not mandatory, but it will be helpful in understanding the tutorial content).

## Mathematical Foundations
Much of the mathematics required for quantum computing comes from linear algebra.

### 1.State Vectors and Dirac Notation
In quantum computing, you will often encounter a symbol like
$$|\psi\rangle$$
This symbol represents the state vector using Dirac notation, and we use it to denote quantum states. State vectors correspond to vectors in linear algebra. Specifically, the ket notation often corresponds to column vectors, while the bra notation corresponds to row vectors.

ket notation is represented as:
$$|\psi\rangle=\left(\begin{array}{c}
\psi_{1} \\
\psi_{2} \\
\psi_{3} \\
\psi_{4} \\
\vdots \\
\psi_{N}
\end{array}\right)$$

bar notation is represented as:
$$\langle\psi|=\left(\begin{array}{llllll}
\psi_{1}^{*}, & \psi_{2}^{*}, & \psi_{3}^{*}, & \psi_{4}^{*}, & \cdots, & \psi_{N}^{*}
\end{array}\right)$$

Here, bar represents the complex conjugate transpose of the ket.

### 2.Inner Product and Tensor Product
#### Inner Product
Similar state vectors (i.e., state vectors with the same dimension, either ket or bra) can be added together.
Furthermore, the inner product can be performed between state vectors$|\varphi\rangle$and$|\psi\rangle$, resulting in a scalar:
$$\langle a \mid b\rangle=c$$
In Python, you can compute the inner product using libraries like PyTorch.
```python
import torch

matrix_a = torch.tensor([[1, 2], [3, 4]])  # 2x2 
matrix_b = torch.tensor([[5, 6], [7, 8]])  # 2x2 

result = torch.matmul(matrix_a, matrix_b)

print(result)
```
```python
import numpy as np

matrix_a = np.array([[1, 2], [3, 4]])  # 2x2 
matrix_b = np.array([[5, 6], [7, 8]])  # 2x2 

result = np.dot(matrix_a, matrix_b)
# or you can use @
# result = matrix_a @ matrix_b

print(result)
```
#### Tensor Product
The tensor product represents an operation where two vector spaces form a larger vector space.The tensor product of vectors $|\alpha\rangle$ and $|\beta\rangle$ is denoted as:
$$|\alpha\rangle \otimes|\beta\rangle=|\alpha\rangle|\beta\rangle=|\alpha \beta\rangle$$
Here are the operational rules for the matrix representation of the tensor product - the Kronecker product:

Let $A$ be an $m \times n$ matrix, and $B$ be a $p \times q$ matrix. The matrix form of $A \otimes B$ is defined as:
$$A \otimes B=\left[\begin{array}{cccc}
A_{11} B & A_{12} B & \cdots & A_{1 n} B \\
A_{21} B & A_{22} & \cdots & A_{2 n} B \\
\vdots & \vdots & \ddots & \vdots \\
A_{m 1} B & A_{m 2} B & \cdots & A_{m n} B
\end{array}\right]$$

In Python, you can also compute the tensor product using libraries like PyTorch.
```python
import numpy as np

matrix_a = np.array([[1, 2], [3, 4]])  # 2x2 
matrix_b = np.array([[5, 6], [7, 8]])  # 2x2 

result = np.kron(matrix_a, matrix_b)

print(result)
```
```python
import torch

matrix_a = torch.tensor([[1, 2], [3, 4]])  # 2x2 
matrix_b = torch.tensor([[5, 6], [7, 8]])  # 2x2 

result = torch.kron(matrix_a, matrix_b)

print(result)
```

### 3.Hermitian Conjugate
The Hermitian conjugate operation is a unary operation, meaning it takes only one input and produces one output.

$$A^{\dagger} = (A^{T})^{*}$$

$$
\begin{array}
  a&b\\\
  c&d
\end{array}^{\dagger}
 = 
\begin{array}
  a^{*}&c^{*}\\\
  b^{*}&d^{*}
\end{array}
$$

We implement this operation using Python.
```python
import numpy as np

matrix = np.array([[1 + 2j, 3 - 4j], [5j, 6]])

transpose_matrix = np.transpose(matrix)

hermitian_conjugate = np.conjugate(transpose_matrix)
```
```python
import torch

matrix = torch.tensor([[1 + 2j, 3 - 4j], [5j, 6]], dtype=torch.complex64)

transpose_matrix = torch.transpose(matrix, 0, 1)

hermitian_conjugate = torch.conj(transpose_matrix)
```

### 4.Hermitian Operator and Unitary Operator
#### Hermitian Operator
$$A^{\dagger} = A$$

#### Unitary Operator
$$A^{\dagger}=A^{-1}$$

***
## Quantum Mechanics Fundamentals
Here, we will focus on understanding the four fundamental postulates of quantum mechanics. These four postulates serve as the foundational rules of quantum computation. With these four postulates, we can abstract the complex processes of quantum computation into matrix operations with specific rules, making it easier for us to comprehend.

### Postulate 1
Any isolated physical system is associated with a Hilbert space. (The system is entirely described by a state vector, which is a unit vector in the system's space.)

### Postulate 2
The evolution of a closed quantum system can be described by unitary transformations.

### Postulate 3
Quantum measurement is described by a set of measurement operators $\{ M_{m} \}$, which act on the state space of the system being measured.
$$P(m) = \langle\psi|M_{m}^{\dagger}M_{m}|\psi\rangle$$

### Postulate 4
The state space of a composite physical system is the tensor product of the state spaces of its individual subsystems.

### The essence of quantum computing
The essence of quantum computing is to simulate the evolution of real quantum states' wavefunctions using quantum circuits and density matrix operations.

***
## Qubits, Quantum Gates, and Quantum Circuits
### 1.Qubits
A classical bit can only have two states, either 0 or 1. If you have two classical bits, there are 4 possible states: 00, 01, 10, and 11. Similarly, with n classical bits, there are 2^n possible states, but at any given time, they can only hold one definite value (state).
However, the situation with quantum bits (qubits) is much more complex. Quantum bits can be in definite states like 0 or 1 (referred to as computational basis states), but they can also exist in superpositions, which means they can be in a combination of both 0 and 1 (referred to as superposition states). We collectively refer to all these possible states as quantum states.
Quantum states can be represented using vectors. For any quantum bit $|\psi\rangle$, we can represent it as:
$$
|\psi\rangle = \begin{pmatrix}
 \alpha\\\beta
\end{pmatrix}
$$
For a single quantum bit (qubit), there are two basis states: 0 and 1. The quantum state represented by a qubit exists within the subspace generated by these two basis states. Therefore, the state of a single qubit is always a linear combination of these two basis states.
$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle 
$$

The expression provided can be seen as a representation of the probabilities of measurement outcomes for a superposition state. α and β represent the probabilities of measurement results collapsing to either $|0\rangle$ or $|1\rangle$.   

When $\alpha = 1$，$\beta = 0$，
$$
|\psi\rangle = 1|0\rangle + 0|1\rangle 
$$
In this case, it's easy to derive:
$$
|0\rangle = \begin{pmatrix}
 1\\0
\end{pmatrix}
$$
Similarly,
$$
|1\rangle = \begin{pmatrix}
 0\\1
\end{pmatrix}
$$
$|0\rangle$ and $|1\rangle$ are two very common basis states in quantum computing. It's important to remember their specific vector representations.

We should also consider the case of multiple qubits. For two qubits, there are 4 basis states: 00, 01, 10, and 11.    
In this case,
$$
|\psi\rangle = \left(\begin{array}{l}
a \\
b \\
c \\
d \\
\end{array}\right)
$$
$$
|\psi\rangle = a|00\rangle + b|01\rangle + c|10\rangle + d|11\rangle 
$$
Similarly, the state of a two-qubit system also exists within the subspace generated by these basis states. Therefore,
$$
|00\rangle=\left(\begin{array}{l}
1 \\
0 \\
0 \\
0
\end{array}\right),|01\rangle=\left(\begin{array}{l}
0 \\
1 \\
0 \\
0
\end{array}\right),|10\rangle=\left(\begin{array}{l}
0 \\
0 \\
1 \\
0
\end{array}\right),|11\rangle=\left(\begin{array}{l}
0 \\
0 \\
0 \\
1
\end{array}\right) \text {. }
$$

### 2.Quantum Gate and Quantum Circuits
#### Quantum Gate

Quantum gates are an abstract representation of the interventions and control we apply to the initial qubit state. Each quantum gate corresponds to a matrix. The following diagram from Wikipedia provides common quantum gate symbols and their corresponding density matrices.

img-wiki

#### Quantum Circuits

Quantum circuits are composed of quantum gates and wires, but they do not necessarily imply that we physically construct circuits using quantum gates during quantum computing. They are an abstract representation depicting the flow of operations in a quantum computation process. Below is a diagram of a quantum computation circuit:

img1

Quantum circuits indicate the number of wires, which represents the number of qubits involved in the computation. In this case, there are four wires, so four qubits are involved in the operation.

However, quantum circuits alone do not provide the results; they only represent the computation's process. Next, we need to use numerical methods based on the initial state and the density matrices corresponding to the quantum gates to calculate the possible outcomes.

Let's proceed with your Python example. First, we need to represent each gate in Python using its corresponding density matrix. 
```python
import torch

def RY(theta):
    U11 = torch.cos(theta / 2)
    U12 = -torch.sin(theta / 2)
    U21 = torch.sin(theta / 2)
    U22 = torch.cos(theta / 2)
    U11 = U11.unsqueeze(1)
    U22 = U22.unsqueeze(1)
    U12 = U12.unsqueeze(1)
    U21 = U21.unsqueeze(1)
    U = torch.cat( (U11,U12,U21,U22) , dim= 1)
    U = U.reshape(2, 2)
    return U

def CNOT():
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], requires_grad=False)

def X():
    return torch.tensor([[0, 1],
                         [1, 0]], requires_grad=False)

def I():
    return torch.eye(2, requires_grad=False)

def PauliZ():
    return torch.tensor([[1, 0], [0, -1]], dtype=torch.float32, requires_grad=True)

```

Then, we build the entire quantum circuit. Here, I've constructed matrices for each layer, and then I perform operations with the initial state. Now, we have to mentione three fundamental operational rules.

#### 1.Composition of quantum gates in the same layer
Operations of quantum gates within the same layer act on different qubits. You want to treat them as a collective operation on the initial state. In this case, you consider all operations in this layer as a composite operation on the entire system. According to the fourth postulate of quantum mechanics, the state space of a composite physical system is the tensor product of the state spaces of its individual subsystems. Therefore, the matrices corresponding to the four quantum gates in this layer should be done a tensor product to generate a composite operation on the initial state.

img2

For the first layer's operation, it should be represented as:
```python
import torch

layer_1 = torch.kron(torch.kron(torch.kron(X(), X()), RY(theta1)), RY(theta2))

```
Now we focus on the third layer, and we will find that we do not have any operations on the third qubit in the third layer. At this time, in order to ensure that the dimensions of the entire system remain unchanged, we need to use I at the third qubit when doing tensor product (identity matrix) placeholder.
The third layer can be expressed as:
```python
import torch

layer_3 = torch.kron(torch.kron(CNOT(), I()), RY(theta))

```

#### 2.Representation of initial state evolution
We can build a complete quantum circuit with the above method, but if we want our initial state to evolve according to the operation we built, we need to use another operation: inner product. Use the initial state and the operation of each layer to do the inner product to get the new density matrix of the system, and then do the inner product with the next layer. In this way, the result of the state evolution can be calculated.
The principle of the inner product operation comes from the basic postulate 2 of quantum mechanics.
All our operations here are unitary operations, and all the operation matrices we construct are unitary matrices. This is the unitary matrix mentioned earlier. If you are sensitive enough, you will find that this means that all these operations are reversible! Yes, this is a very important feature of quantum computing, all operations are reversible. As for why this is the case, you can look up relevant information and try to prove it mathematically.

#### 3.Measurement of Quantum States
According to the basic postulate 4 of quantum mechanics, we can measure our results after the calculation is completed.
```python
U = torch.matmul(layer_5, torch.matmul(layer_4, torch.matmul(layer_3, torch.matmul(layer_2, layer_1))))
alpha = torch.matmul(U, input_state.float())
M = torch.kron(torch.kron(torch.kron(I(), I()), I()), PauliZ())
conjugate_alpha = torch.conj(alpha.float())
transpose_alpha = torch.transpose(conjugate_alpha, 0, 1)
output_value = torch.matmul(torch.matmul(transpose_alpha.float(), M.float()), alpha.float())

```

These three operation rules come from the three basic postulates of quantum mechanics, so there is postulate 1 that has not been mentioned.
Postulate 1 gives a method to characterize quantum states in a set space. On a Bloch sphere, we can accurately describe the position of the state through the three phases of the quantum state, as shown in the figure:

img3

Now that we have built a quantum circuit and understand the basic content we need to know to perform quantum computing, let's expand a little and see what else we can do on this basis.

***
## Variational Quantum Algorithm, VQA
Variational quantum algorithm uses a classical optimizer to train a parameterized quantum circuit. It is somewhat like the natural kind of machine learning in quantum computing.

This is also the basis of quantum machine learning.

I will put the built quantum circuit and parameterized training process in another folder of this project. If you are interested, you can run it to see the effect.



