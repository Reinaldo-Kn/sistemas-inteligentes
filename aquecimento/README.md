### 1. O que representa a "reta" no gráfico?

Representa a **fronteira de decisão** aprendida pelo perceptron. Essa fronteira separa os dados de entrada em duas regiões:

- Conjunto lógico verdadeiro, região na cor vermelha 
- Conjunto lógico falso, região na cor azul

Ela é resultado de um classificador binário, ou é seja, o perceptron tenta encontrar uma linha que separe os dados de entrada em duas classes distintas.

---

### 2. É possível atingir sucesso no treinamento com a estrutura atual da rede para a função XOR?

Não é possível.

A função lógica XOR **não é linearmente separável**, e o perceptron é um classificador linear. Logo, não existe uma linha que consiga dividir corretamente as classes `0` e `1` no espaço de entrada.

Não foi possível nem simular o comportamento do XOR, pois o perceptron não consegue aprender a função XOR.

Para resolver o problema da XOR, seria necessário utilizar uma **rede neural com pelo menos uma camada oculta** (Multi Layer Perceptron - MLP).
Como vemos na imagem

![MLP](https://web.fs.uni-lj.si/lasin/wp-content/include-me/neural/nn04_mlp_xor/nn04_mlp_xor_01.png)

Autoria da Imagem: [web.fs.uni-lj.si](https://web.fs.uni-lj.si/lasin/wp-content/include-me/neural/nn04_mlp_xor/)