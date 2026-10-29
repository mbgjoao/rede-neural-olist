
"""Passo 1 do enunciado: a rede da playlist de referência.

Reimplementação da classe NnModel da playlist "Rede neural do zero em
Python" (canal Machine Learning para humanos), feita a partir das
transcrições automáticas dos vídeos #01 a #08. Os nomes (NnModel,
hidden_neurons, model_dict, forward, loss, backpropagation, fit) seguem os
do vídeo para facilitar a comparação.

  * 1 camada oculta com tangente hiperbólica (vídeo #03);
  * saída softmax com um neurônio por classe (vídeos #03 e #05);
  * entropia cruzada média (vídeo #04);
  * retropropagação com delta2 = softmax - 1 na classe correta (vídeo #05);
  * descida do gradiente com todos os exemplos a cada época (vídeo #06).

Limitação: o código exibido na tela não foi lido diretamente. Nomes
internos, a semente e o formato das mensagens podem diferir do original.
Na transcrição do vídeo #05 não aparece divisão pelo número de exemplos m;
esta versão soma os gradientes, o que equivale a usar o gradiente médio com
taxa de aprendizado lr * m. A única adição é numérica: a softmax subtrai o
maior valor de cada linha antes do exp (o resultado é o mesmo, mas exp não
estoura com valores grandes).

A rede customizada para os dados da Olist está em rede_neural.py.

Demonstração, como no vídeo #01 (make_moons com 500 pontos):
    python src/rede_referencia.py
"""

import numpy as np


class NnModel:
    def __init__(self, X, y, hidden_neurons=10, output_neurons=2, seed=0):
        np.random.seed(seed)
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)
        self.input_neurons = self.X.shape[1]
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        # Pesos aleatórios divididos pela raiz do número de entradas da
        # camada; vieses começam em zero (vídeo #02).
        self.W1 = (np.random.randn(self.input_neurons, hidden_neurons)
                   / np.sqrt(self.input_neurons))
        self.B1 = np.zeros((1, hidden_neurons))
        self.W2 = (np.random.randn(hidden_neurons, output_neurons)
                   / np.sqrt(hidden_neurons))
        self.B2 = np.zeros((1, output_neurons))
        self.model_dict = {"W1": self.W1, "b1": self.B1,
                           "W2": self.W2, "b2": self.B2}

        # Guardados no forward para serem usados na retropropagação.
        self.z1 = 0
        self.f1 = 0
        self.z2 = 0

    def forward(self, X):
        """Probabilidades de cada classe, formato (m, output_neurons)."""
        self.z1 = X.dot(self.W1) + self.B1
        self.f1 = np.tanh(self.z1)
        self.z2 = self.f1.dot(self.W2) + self.B2
        # Subtrair o máximo da linha não muda a softmax e evita exp(710+) = inf.
        exp_values = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        softmax = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return softmax

    def loss(self, softmax):
        """Entropia cruzada: média de -log da probabilidade da classe correta."""
        predictions = softmax[range(self.y.shape[0]), self.y]
        log_prob = -np.log(predictions)
        return np.mean(log_prob)

    def gradients(self, softmax):
        """Derivadas da perda (somada nos exemplos) em relação a W e b."""
        delta2 = np.copy(softmax)
        delta2[range(self.y.shape[0]), self.y] -= 1
        dW2 = (self.f1.T).dot(delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = delta2.dot(self.W2.T) * (1 - np.power(np.tanh(self.z1), 2))
        dW1 = (self.X.T).dot(delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def backpropagation(self, softmax, learning_rate):
        grads = self.gradients(softmax)
        # "+= -lr * d" altera os arrays no lugar, então model_dict, que
        # aponta para os mesmos arrays, também é atualizado.
        self.W1 += -learning_rate * grads["W1"]
        self.B1 += -learning_rate * grads["b1"]
        self.W2 += -learning_rate * grads["W2"]
        self.B2 += -learning_rate * grads["b2"]

    def fit(self, epochs, lr, verbose=True):
        """Treina por um número fixo de épocas e retorna as previsões finais."""
        for epoch in range(epochs):
            outputs = self.forward(self.X)
            loss = self.loss(outputs)
            self.backpropagation(outputs, lr)

            prediction = np.argmax(outputs, axis=1)
            correct = (prediction == self.y).sum()
            accuracy = correct / self.y.shape[0]

            if verbose and int((epoch + 1) % max(1, epochs // 10)) == 0:
                print(f"Época: [{epoch + 1} / {epochs}]  "
                      f"Acurácia: {accuracy:.3f}  Perda: {loss.item():.4f}")
        return prediction


if __name__ == "__main__":
    from sklearn.datasets import make_moons

    # Vídeo #01: make_moons com 500 observações. Aqui com ruído 0,1 para
    # que o problema não fique trivial.
    X, y = make_moons(n_samples=500, noise=0.1, random_state=0)
    # Vídeo #06: 10 neurônios e taxa de aprendizado 0,001 (o vídeo começa
    # com 100 épocas; aqui são 1000 para chegar perto da convergência).
    for epocas in (100, 1000):
        modelo = NnModel(X, y, hidden_neurons=10, output_neurons=2)
        print(f"--- {epocas} épocas ---")
        modelo.fit(epochs=epocas, lr=0.001, verbose=epocas == 100)
        acuracia = (modelo.forward(X).argmax(axis=1) == y).mean()
        print(f"Acurácia final no make_moons com {epocas} épocas: "
              f"{acuracia:.3f}\n")
