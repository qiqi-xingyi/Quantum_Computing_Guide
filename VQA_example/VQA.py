# -*- coding: utf-8 -*-
# time: 2023/5/31 16:15
# file: VQA.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义量子门操作函数
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




def partial_trace(rho, keep, dims, optimize=False):

    keep = np.asarray(keep)  # 将 `keep` 数组转换为 NumPy 数组，方便后续处理。
    dims = np.asarray(dims)  # 将 `dims` 数组转换为 NumPy 数组，方便后续处理。
    Ndim = dims.size  # 复合量子系统的子系统数目。
    Nkeep = np.prod(dims[keep])  # 计算偏迹后保留子系统的维度。
    # print("keep:" , Nkeep)

    # 创建 einsum 缩并的索引列表。
    idx1 = [i for i in range(Ndim)]  # 缩并的第一部分的索引列表。
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]  # 缩并的第二部分的索引列表。

    # 将输入的密度矩阵进行形状变换，为 einsum 缩并做准备。
    rho_a = rho.reshape(np.tile(dims, 2))

    # 使用 einsum 缩并计算偏迹。
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=False)

    # 将结果矩阵重新调整为迹掉子系统后的期望形状。
    rho_a = rho_a.reshape(Nkeep, Nkeep)

    return rho_a


def compute_entropy(input_rho):

    rho_tensor = torch.tensor(input_rho)

    #计算纠缠熵
    evals = torch.linalg.eigvalsh(rho_tensor)

    entropy = -torch.sum(evals * torch.log2(evals))

    return  entropy



class QuantumCircuit(nn.Module):
    def __init__(self):
        super(QuantumCircuit, self).__init__()
        # 定义可训练参数
        self.theta1 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta3 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta4 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta5 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta6 = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, input_state):
        # 定义量子网络
        layer_1 = torch.kron(torch.kron(torch.kron(X(), X()), RY(self.theta1)), RY(self.theta2))
        layer_2 = torch.kron(torch.kron(RY(self.theta3), RY(self.theta4)), CNOT())
        layer_3 = torch.kron(torch.kron(CNOT(), I()), RY(self.theta5))
        layer_4 = torch.kron(torch.kron(torch.kron(I(), RY(self.theta6)), I()), I())
        layer_5 = torch.kron(torch.kron(I(), I()), CNOT())
        layer_5 = layer_5.reshape(2, 2, 2, 2, 2, 2, 2, 2)
        layer_5 = torch.transpose(torch.transpose(layer_5, 1, 2), 5, 6)
        layer_5 = layer_5.reshape(16, 16)
        U = torch.matmul(layer_5, torch.matmul(layer_4, torch.matmul(layer_3, torch.matmul(layer_2, layer_1))))
        alpha = torch.matmul(U, input_state.float())
        M = torch.kron(torch.kron(torch.kron(I(), I()), I()), PauliZ())
        conjugate_alpha = torch.conj(alpha.float())
        transpose_alpha = torch.transpose(conjugate_alpha, 0, 1)
        output_value = torch.matmul(torch.matmul(transpose_alpha.float(), M.float()), alpha.float())

        return output_value

quantum_circuit = QuantumCircuit()

if __name__ == '__main__':

    quantum_circuit.train()

    # 把图片编码成量子态
    BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    input_states = []
    for image in BAS:
        quantum_state = torch.zeros(4, 2, 1)
        for i, pixel in enumerate(image):
            quantum_state[i][int(pixel)][0] = 1
        input_states.append(quantum_state)

    # 构建输入态
    states = []
    for input_state in input_states:
        state = torch.kron(input_state[0], torch.kron(input_state[1], torch.kron(input_state[2], input_state[3])))
        states.append(state)

    # 数据集标签
    label = [-1.0, -1.0, 1.0, 1.0]
    label_tensors = [torch.tensor(element, dtype=torch.float32) for element in label]

    # 设置超参数
    learning_rate = 0.2
    num_epochs = 100

    # 定义优化器
    optimizer = optim.SGD(quantum_circuit.parameters(), lr=learning_rate)

    loss_fn = nn.MSELoss()

    # 迭代训练
    for epoch in range(num_epochs):
        running_loss = 0.0
        for input_state, label_tensor in zip(states, label_tensors):
            # 将梯度归零
            optimizer.zero_grad()
            # 前向传播
            output = quantum_circuit(input_state)
            # print("output:", output)
            label = label_tensor.view(output.shape)
            # 计算损失值
            # loss = torch.mean(torch.abs(output - label))
            loss = loss_fn(output, label)
            loss.backward()

            # for name, param in quantum_circuit.named_parameters():
            #     # if param.grad is not None:
            #     print("the grad:", name, param.grad)
            #
            # print("*********************************")

            # 更新参数
            optimizer.step()
            # 累计损失值
            running_loss += loss.item()


        # 打印平均损失值
        average_loss = running_loss / len(states)
        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

    #打印参数值
    print('####################################################################')
    for param in quantum_circuit.parameters():
        print('param:' , param)

    #############################################################################################

    # 设置模型为推理模式
    quantum_circuit.eval()

    # 把图片编码成量子态
    BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    input_states = []
    for image in BAS:
        quantum_state = torch.zeros(4, 2, 1)
        for i, pixel in enumerate(image):
            quantum_state[i][int(pixel)][0] = 1
        input_states.append(quantum_state)

    # 构建输入态
    states = []
    for input_state in input_states:
        state = torch.kron(input_state[0], torch.kron(input_state[1], torch.kron(input_state[2], input_state[3])))
        states.append(state)

    print('####################################################################')
    # 进行推理
    for input_state in states:
        # 前向传播
        output = quantum_circuit(input_state)

        # 打印输出结果

        print("Measurement result:", output)


    print('####################################################################')
    # 获取训练后的参数

    model = QuantumCircuit()

    # 固定参数
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    layer_1 = torch.kron(torch.kron(torch.kron(X(), X()), RY(model.theta1)), RY(model.theta2))
    layer_2 = torch.kron(torch.kron(RY(model.theta3), RY(model.theta4)), CNOT())
    layer_3 = torch.kron(torch.kron(CNOT(), I()), RY(model.theta5))
    layer_4 = torch.kron(torch.kron(torch.kron(I(), RY(model.theta6)), I()), I())
    layer_5 = torch.kron(torch.kron(I(), I()), CNOT())
    layer_5 = layer_5.reshape(2, 2, 2, 2, 2, 2, 2, 2)
    layer_5 = torch.transpose(torch.transpose(layer_5, 1, 2), 5, 6)
    layer_5 = layer_5.reshape(16, 16)
    U = torch.matmul(layer_5, torch.matmul(layer_4, torch.matmul(layer_3, torch.matmul(layer_2, layer_1))))

    state_for_e = states[0]

    alpha = torch.matmul(U , state_for_e)

    # print(alpha.size())

    alpha_conj = torch.conj(alpha)  # 复共轭
    alpha_conj_t = torch.transpose(alpha_conj , 0 , 1)  # 转置

    rho = torch.kron(alpha , alpha_conj_t)
    # print(rho.size())

    evals = torch.linalg.eigvalsh(rho)
    # print(evals)
    # for eval in evals:

    entropy = -torch.sum(evals * torch.log2(evals))
    # print("the entropy is:" , entropy)


    #密度函数 rho
    rho_array = rho.numpy()

    qubit_1_rho = partial_trace(rho_array , [1,2,3] , [2,2,2,2])
    qubit_1_rho = torch.tensor(qubit_1_rho)
    evals_1 = torch.linalg.eigvalsh(qubit_1_rho)
    print("evals_1:", evals_1)
    print("qubit_1 is:" , compute_entropy(qubit_1_rho))
    print("sum of eigenvalue_1:", torch.sum(evals_1))

    qubit_2_rho = partial_trace(rho_array, [0, 2, 3], [2, 2, 2, 2])
    print("qubit_2 is:" , compute_entropy(qubit_2_rho))
    qubit_2_rho = torch.tensor(qubit_2_rho)
    evals_2 = torch.linalg.eigvalsh(qubit_2_rho)
    print("evals_2:", evals_2)
    print("sum of eigenvalue_2:", torch.sum(evals_2))

    qubit_3_rho = partial_trace(rho_array, [0, 1, 2], [2, 2, 2, 2])
    print("qubit_3 is:" , compute_entropy(qubit_3_rho))
    qubit_3_rho = torch.tensor(qubit_3_rho)
    evals_3 = torch.linalg.eigvalsh(qubit_3_rho)
    print("evals_3:", evals_3)
    print("sum of eigenvalue_3:", torch.sum(evals_3))

    qubit_4_rho = partial_trace(rho_array, [0, 1, 2], [2, 2, 2, 2])
    print("qubit_4 is:" , compute_entropy(qubit_4_rho))
    qubit_4_rho = torch.tensor(qubit_4_rho)
    evals_4 = torch.linalg.eigvalsh(qubit_4_rho)
    print("evals_4:", evals_4)
    print("sum of eigenvalue_4:", torch.sum(evals_4))

    tensor_of_matrices = torch.stack([evals_1 , evals_2 , evals_3 ,evals_4])
    sum_of_all_elements = torch.sum(tensor_of_matrices)

    print("sum:" , sum_of_all_elements)