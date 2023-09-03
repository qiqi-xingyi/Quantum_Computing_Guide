# Quantum Computing Guide
A quantum computing tutorial for engineering students starting from scratch.

本教程会从基础的数学和量子力学基础开始，以非常简洁明了的语言帮助你理解量子计算的基本概念并告诉你如何搭建量子线路、如何进行数值模拟。

这份指南需要你具备一点基础的线性代数知识（例如：你需要知道什么是矩阵）、一点Python的语法（这并不是必须的，但是有助于理解教程的内容）。

***
## 数学基础
量子计算需要用到的数学大部分都来自于线性代数。

### 1.态矢和狄拉克算符
在量子计算中我们会经常看见一个符号
$$|\psi\rangle$$
这个符号代表的是狄拉克算符的态矢，我们用它来表示量子态。态矢与线性代数中的矢量相对应。具体的，右矢往往对应于列矢量，左矢往往对应于行矢量。      
ket表示为：
$$|\psi\rangle=\left(\begin{array}{c}
\psi_{1} \\
\psi_{2} \\
\psi_{3} \\
\psi_{4} \\
\vdots \\
\psi_{N}
\end{array}\right)$$
 
bar表示为：
$$\langle\psi|=\left(\begin{array}{llllll}
\psi_{1}^{*}, & \psi_{2}^{*}, & \psi_{3}^{*}, & \psi_{4}^{*}, & \cdots, & \psi_{N}^{*}
\end{array}\right)$$
在这里，bar是ket的共轭转置。

### 2.内积与张量积
#### 内积
同类态矢（即同维右矢或者左矢）间可以做加法。   
此外，态矢$|\varphi\rangle$和态矢$|\psi\rangle$间可以做内积,得到的是一个数：
$$\langle a \mid b\rangle=c$$
在Python中内积计算可以使用pytorch等库完成：
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
#### 张量积
张量积代表两个向量空间形成一个更大向量空间的运算。  
向量$|\alpha\rangle$和向量$|\beta\rangle$的张量积为：
$$|\alpha\rangle \otimes|\beta\rangle=|\alpha\rangle|\beta\rangle=|\alpha \beta\rangle$$
下面给出张量积的矩阵表示的运算规则-克罗内科积   
设$A$是$m \times n$的矩阵， $B$是$p \times q$的矩阵, $A\bigotimes B$的矩阵形式定义为
$$A \otimes B=\left[\begin{array}{cccc}
A_{11} B & A_{12} B & \cdots & A_{1 n} B \\
A_{21} B & A_{22} & \cdots & A_{2 n} B \\
\vdots & \vdots & \ddots & \vdots \\
A_{m 1} B & A_{m 2} B & \cdots & A_{m n} B
\end{array}\right]$$

在Python中张量积计算也可以使用pytorch等库完成：
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

### 3.厄米共轭
厄米共轭运算是一种一元运算（即只输入一个东西，出来一个东西）。
$$A^{\dagger} = (A^{T})^{*}$$
$$\begin{bmatrix}
  a&b\\
  c&d
\end{bmatrix}^{\dagger}
=
\begin{bmatrix}
  a^{*}&c^{*}\\
  b^{*}&d^{*}
\end{bmatrix}$$
Python实现：
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

### 4.厄米算符和幺正算符
#### 厄米算符
$$A^{\dagger} = A$$

#### 幺正算符
$$A^{\dagger}=A^{-1}$$

***
## 量子力学基础
在这里我们重点掌握量子力学的四个基本公设即可。四个基本公设是量子计算的底层规则，有这四个公设我们就可以把量子计算的复杂过程抽象成有一定特殊规则的矩阵计算，这样可以方便我们理解。
### 公设1
任意一个孤立的物理系统都与希尔伯特空间相联系。     
（系统完全由状态向量描述，它是系统空间里面的一个单位向量）
### 公设2
封闭量子系统的演化可用酉变换描述
### 公设3
量子测量由一组测量算子$\left \{ M_{m}\right \}$描述，这些算子作用在被测量系统的状态空间上。
$$P(m) = \langle\psi|M_{m}^{\dagger}M_{m}|\psi\rangle$$
### 公设4
复合物理系统的状态空间是分物理系统的状态空间的张量积

### 量子计算的本质
量子计算的本质就是通过量子线路、密度矩阵运算的方式模拟真实的量子态波函数演化的过程。

***
## 量子比特、量子门和量子线路
### 1.量子比特
一个数字比特只有两种状态，要么是0要么是1。如果是两个数字比特则有4种状态00，01，10和11 。以此类推$n$个数字比特就有$2^{n}$种状态，但在任何时间都只能是一个确定值（状态）。

而量子比特的情况则要复杂得多，量子比特不仅和数字比特一样可以是像0或1 这样确定的状态（称为计算基态），还有可能是0和1两种状态的叠加（称为叠加态）。我们将各种可能的状态统称为量子态。

量子态可以用向量表示，对于任意量子比特$|\psi\rangle$,我们都可以表示为：
$$
|\psi\rangle = \begin{pmatrix}
 \alpha\\\beta
\end{pmatrix}
$$
并且对于一个量子比特的情况，我们有两种基态：0和1，同时量子比特表示的量子态存在于这两种基态生成的子空间里，因此，单量子比特的状态一定是这两种基态的线性组合。
$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle 
$$
这里的表达可以看成是一种对叠加态测量结果的概率表示，$\alpha$和$\beta$表示测量结果坍缩在$|0\rangle$或$|1\rangle$的可能性。  
当$\alpha = 1$，$\beta = 0$时，
$$
|\psi\rangle = 1|0\rangle + 0|1\rangle 
$$
此时很容易得到：
$$
|0\rangle = \begin{pmatrix}
 1\\0
\end{pmatrix}
$$
同理，
$$
|1\rangle = \begin{pmatrix}
 0\\1
\end{pmatrix}
$$
$|0\rangle$和$|1\rangle$是量子计算中非常常见的两种基态，请一定要记住他们的具体向量表示。

我们还应该考虑多比特的情况。两个量子比特有4个基态：00，01，10 和 11。
此时
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
同时两个量子比特情况的状态也一定存在在这个子空间中，因此
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
量子门是一种抽象表示，它表示我们对初态qubit的一系列干预和控制，每个量子门都对应一个矩阵。下图来自维基百科，给出了常用的量子门符号以及对应的密度矩阵。

img-wiki

#### Quantum Circuits
Quantum Circuits是由Quantum Gate和wire组成的一种Circuits，但是并不代表我们在做量子计算的时候需要真的通过Quantum Gate来搭建线路。它是一种抽象的表示，表示的是进行一个量子计算的过程中操作的流程。下面是一个量子计算的线路图：

img1


Quantum Circuits的wire数目代表的是参与运算的qubits数量，这里有四条wire，所以有四位qubits参与了运算。

但是仅有Quantum Circuits我们还是无法估计得到的结果，我们目前仅仅是通过Quantum Circuits表示了此次计算的流程。接下来我们需要根据初态和量子门对应的密度矩阵通过数值方法计算可能会出现的结果。

我用python代码做一个示例

首先我们需要根据量子门对应的密度矩阵在python中表示每个门
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
然后搭建整个量子线路，这里我构建了每一层的矩阵，然后再用初态进行运算，这里就不得不提到三个基本的运算规则：   

#### 1.同一层量子门的复合
同一层的量子门之间是作用在不同qubit上的操作，我想要把他们当成对初态的一个整体操作来看待，此时我将这层的所有操作视作一个对复合系统的操作，根据量子力学基本公设4，复合物理系统的状态空间是分物理系统的状态空间的张量积，因此这一层的四个量子门对应的矩阵应该做张量积才能生成对初态的一个复合操作。

img2

对于第一层的操作，应该表示为：
```python
import torch

layer_1 = torch.kron(torch.kron(torch.kron(X(), X()), RY(theta1)), RY(theta2))

```
现在我们关注第三层，会发现我们在第三层对第三位qubit没有任何操作，此时为了保证整个系统的维度不变，我们需要在做张量积时在第三位qubit处用I（单位矩阵）占位。
第三层可以表示为：
```python
import torch

layer_3 = torch.kron(torch.kron(CNOT(), I()), RY(theta))

```

#### 2.初态演化的表示
我们用上述方法可以构建一个完整的量子线路，但是想让我们的初态按照我们构建好的操作发生演化就得用到另一个操作：内积。用初态和每一层的操作做内积，得到系统新的密度矩阵，然后再和下一层做内积，用这样的方式就可以计算态演化后的结果。
内积操作的原理来源于量子力学基本公设2。   
在这里我们的操作全部都是酉操作，我们的构建的操作矩阵全部都是酉矩阵。也就是之前提到过的幺正矩阵。如果你足够敏感，你会发现这代表这这些操作全部都是可逆的！没错，这就是量子计算的一个非常重要的特性，所有的操作全部可逆。至于为什么是这样，您可以查阅相关资料并尝试用数学证明。

#### 3.量子态的测量
根据量子力学基本公设4，我们可以在计算结束后测量我们的结果。
```python
U = torch.matmul(layer_5, torch.matmul(layer_4, torch.matmul(layer_3, torch.matmul(layer_2, layer_1))))
alpha = torch.matmul(U, input_state.float())
M = torch.kron(torch.kron(torch.kron(I(), I()), I()), PauliZ())
conjugate_alpha = torch.conj(alpha.float())
transpose_alpha = torch.transpose(conjugate_alpha, 0, 1)
output_value = torch.matmul(torch.matmul(transpose_alpha.float(), M.float()), alpha.float())

```

这三个运算规则来自于三个量子力学基本公设，那么还有公设1没有提到。
公设1给出了一种在集合空间中表征量子态的方法，在一个布洛赫球面上，我们可以通过量子态的三个相位准确刻画态的位置，如图所示：

img3

现在我们已经搭建了量子线路，也了解了进行量子计算需要知道的基础内容，接下来我们稍微拓展一下，看看在此基础上我们还能够做什么。

***
## 变分量子算法, VQA
变分量子算法就是用一个经典优化器（classical optimizer）来训练一个含参量子线路（quantum circuit）它有些像是机器学习在量子计算中的自然类

这也是量子机器学习的基础。

我将搭建好的量子线路以及参数化后的训练过程放在这个项目的另一个文件夹中，如果您感兴趣可以运行看看效果。




























