import numpy as np
import matplotlib.pyplot as plt

def f(u):
    # Função de ativação
    return 1 if u >= 0 else -1

def findOutput(data, w):
    # Calcular a saída do perceptron
    u = np.dot(w, data)  # Usando produto escalar
    return f(u)

def train_perceptron(p, d, c, d_error, max_iter):
    w = np.random.rand(len(p[0]))  # Inicialização aleatória dos pesos
    iter = 0
    errors = []

    while iter < max_iter:
        error = 0
        for i in range(len(p)):
            o = findOutput(p[i], w)
            error += ((d[i] - o) ** 2) / 2
            learningSignal = c * (d[i] - o)
            for k in range(len(p[i])):
                w[k] += learningSignal * p[i][k]
        
        errors.append(error)
        print(f"Iteração {iter}: Erro = {error}, Pesos = {w}")
        
        if error < d_error:
            print('Convergência alcançada.')
            break
        iter += 1

    return w, errors

def validate(data, w):
    print("Resultados da validação:")
    for x in data:
        print(f"Entrada: {x}, Saída: {findOutput(x, w)}")

# Configurações para a porta AND
p = [[1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]]  # Conjunto de dados
d = [1, -1, -1, -1]  # Saídas desejadas
c = 0.1  # Taxa de aprendizado
d_error = 0.01  # Erro desejado
max_iter = 1000  # Máximo de iterações

# Treinamento do perceptron
weights, error_history = train_perceptron(p, d, c, d_error, max_iter)

# Validação
validate(p, weights)

# Visualização do erro
plt.plot(error_history)
plt.xlabel('Iterações')
plt.ylabel('Erro')
plt.title('Convergência do Perceptron')
plt.show()
