from scipy.optimize import basinhopping, minimize
import scipy.signal
import numpy as np
import math
import matplotlib.pyplot as plt

# Simula um sistema discreto
def simulateSystem(B, A, kstart, kend, r, y, e, u):
    e = e.copy()
    y = y.copy()
    u = u.copy()

    nb = len(B) - 1             # number of zeros
    na = len(A) - 1             # number of poles

    # Simulate
    for k in range(kstart, kend):
        # adiciona erro
        y[k] = np.dot(B, u[k-1:k-1-(nb+1):-1]) - np.dot(A[1:], y[k-1:k-1-na:-1]) 
        
        # Adicionando erro de modelo 
        y[k] += 0.2e-10 * np.random.randn()  
        
        # error
        e[k] = r[k] - y[k]

    return y, e

# Calcula o funcional de controle que minimiza o erro
# de rastreamento (tracking) e energia de controle.
#
# L é o parâmetro lambda para penalizar a energia de controle.
# L=0 torna o controlador mais agressivo possível.
#
# Ver: https://en.wikipedia.org/wiki/Model_predictive_control
def Jcost(r, y, u, L):
    # Variação de controle (delta U)
    du = u[1:] - u[:-1]

    Jtracking = np.sum((r - y)**2)
    Jcontrol = L*np.sum(du**2)

    J = Jtracking + Jcontrol

    return J

# Calcula a função objetivo a ser minimizada.
# Simula o sistema dentro de uma janela (kstart, kend)
# e calcula o funcional ótimo (J).
def fobj(B, A, kstart, kend, r, y, e, u, L):
    y, _ = simulateSystem(B, A, kstart, kend, r, y, e, u)    

    return Jcost(r, y, u, L)

def main():
    # Função de transferência contínua - modelo do controlador ótimo
    P = scipy.signal.TransferFunction([25], [3, 1])

    # Discretizada
    Ts = 0.1                    # time step
    Pd = P.to_discrete(Ts)

    B = Pd.num                  # zeros
    A = Pd.den                  # poles
    nb = len(B) - 1             # number of zeros
    na = len(A) - 1             # number of poles

    # Parâmetros da simulação
    tf = 5

    slack = np.amax([na, nb]) + 1 # slack for negative time indexing of arrays
    kend = math.ceil(tf/Ts) + 1   # end of simulation in discrete time
    kmax = kend + slack           # total simulation array size

    y = np.zeros(kmax)
    u = np.zeros(kmax)
    e = np.zeros(kmax)
    r = np.ones(kmax)
    r[:kmax//4] = 1
    r[kmax//4:2*kmax//4] = 5
    r[2*kmax//4:3*kmax//4] = 2
    r[3*kmax//4:] = 3

    # Testar diferentes valores de L para observar a influência na tolerância a erros
    L_values = [0, 0.1, 1, 10]  
    
    results = {}
    
    for L in L_values:
        y_temp = np.zeros(kmax)
        u_temp = np.zeros(kmax)
        e_temp = np.zeros(kmax)
        
        # Simulação principal com cálculo do controle ótimo
        for k in range(slack, kmax):
            # Simula o sistema
            y_temp[k] = np.dot(B, u_temp[k-1:k-1-(nb+1):-1]) - np.dot(A[1:], y_temp[k-1:k-1-na:-1])
            y_temp[k] += 0.2e-10 * np.random.randn()  # Mesmo erro de modelo durante a simulação
            
            # Erro
            e_temp[k] = r[k] - y_temp[k]
            
            # Controle ótimo baseado em modelo
            f = lambda du : fobj(B, A, kstart=k, kend=kmax, r=r, y=y_temp, e=e_temp, 
                                u=np.r_[u_temp[:k], u_temp[k-1] + du], L=L)
            x0 = np.zeros_like(u_temp[k:])
            
            # Aplicando basinhopping para otimização global
            minimizer_kwargs = {"method": "BFGS", "options": {"disp": False, "gtol": 1e-10}}
            #res = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs, niter=10)
            #du = res.x[0]
            
            # Primeiro faz a busca global com basinhopping
            res_global = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs, niter=10)

            # Depois refina localmente com minimize
            res_local = minimize(f, res_global.x, method='BFGS', options={'gtol': 1e-10, 'disp': False})

            # Usa o melhor entre os dois
            if res_local.fun < res_global.fun:
                du = res_local.x[0]
            else:
                du = res_global.x[0]



            u_temp[k] = u_temp[k-1] + du

        # ignore empty slack
        results[L] = {
            'y': y_temp[slack:],
            'u': u_temp[slack:],
            'e': e_temp[slack:],
            'J': Jcost(r[slack:], y_temp[slack:], u_temp[slack:], L)
        }
        print(f'L = {L}, J cost: {results[L]["J"]}')

    # Plot time response for different L values
    t = np.arange(0, tf + Ts, Ts)
    fig, ax = plt.subplots(2, sharex=True, figsize=(10, 8))

    # Plot outputs
    ax[0].plot(t, r[slack:], 'k--', label='Reference')
    for L, data in results.items():
        ax[0].plot(t, data['y'], label=f'L={L}, J={results[L]["J"]:.4e}')
    ax[0].set_ylabel('y(t)')
    ax[0].legend()
    ax[0].set_title('Resposta do Sistema com Erros de Modelo para Diferentes Valores de L')

    # Plot control signals
    for L, data in results.items():
        ax[1].plot(t, data['u'], label=f'L={L}, J={results[L]["J"]:.4e}')
    ax[1].set_ylabel('u(t)')
    ax[1].legend()
    plt.xlabel('t (s)')
    plt.xlim([0, tf])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
