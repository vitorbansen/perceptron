import numpy as np
import math

def f(u):
    return (2 / (1 + math.exp(-u))) - 1

def findOutput(data, w):
    u = np.dot(w, data)  # Usando np.dot para simplificar a soma ponderada
    lamb = 0.10
    return f(lamb * u)

def train_delta(p, d, learning_rate, desired_error, max_iter):
    w = np.random.rand(len(p[0]))  # Inicialização aleatória dos pesos
    iter = 0

    while True:
        error = 0
        for i in range(len(p)):
            o = findOutput(p[i], w)
            error += 0.5 * (d[i] - o) ** 2.0
            delta = (d[i] - o) * (1 - o ** 2)  # Derivada da função de ativação
            for k in range(len(p[i])):
                w[k] += learning_rate * delta * p[i][k]
                print(f'Peso atualizado: {w}')  # Imprimindo os pesos

        iter += 1
        print(f'Iteração {iter}, Erro: {error}, Pesos: {w}')
        
        if error < desired_error or iter >= max_iter:
            print('Número de iterações:', iter)
            break

    return w

# Testando com a porta AND
print("Treinando perceptron para a porta AND:")
p_and = [[1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]]
d_and = [1, -1, -1, -1]  # Saídas desejadas para a porta AND

learning_rate = 0.5
desired_error = 0.01
max_iter = 1000

w_and = train_delta(p_and, d_and, learning_rate, desired_error, max_iter)

# Testando a saída para a porta AND
for input_data in p_and:
    print(f'Saída para {input_data[:-1]}: {findOutput(input_data, w_and)}')

# Testando com a porta XOR (não linearmente separável)
print("\nTentando a porta XOR (deve falhar):")
p_xor = [[1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]]
d_xor = [-1, 1, 1, -1]  # Saídas desejadas para a porta XOR

w_xor = train_delta(p_xor, d_xor, learning_rate, desired_error, max_iter)

# Testando a saída para a porta XOR
for input_data in p_xor:
    print(f'Saída para {input_data[:-1]}: {findOutput(input_data, w_xor)}')
