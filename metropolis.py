import numpy as np
import matplotlib.pyplot as plt
import pickle
from numba import jit

from wang import (energy_ising, # calcula a energia da rede na configuração atual
                  neighbours,   # computa a matriz de vizinhança da rede
                  random_energy_state # cria uma configuração aleatória
                 )

@jit(nopython=True) 
def expos(beta): 
    """
    Cria um vetor com o cálculo das exponenciais utilizadas.
    Isso economiza tempo nesse tipo de operação
    """
    ex = np.zeros(5,dtype=np.float32) 
    ex[0]=np.exp(8.0*beta) 
    ex[1]=np.exp(4.0*beta)
    ex[2]=0.0 
    ex[3]=np.exp(-4.0*beta) 
    ex[4]=np.exp(-8.0*beta) 
    return ex

@jit(nopython=True)
def mcstep(beta,s,viz,ener,mag):
    """
    Realiza um passo de Monte Carlo na rede s passada, à temperatura beta,
    com vizinhos viz
    """
    N = len(s) 
    ex = expos(beta) 
    for i in range(N): 
        # Soma dos vizinhos
        h = s[viz[i,0]]+s[viz[i,1]]+s[viz[i,2]]+s[viz[i,3]]
        
        # Delta de energia se flipar esse sítio
        de = int(s[i]*h*0.5+2)
        
        if np.random.random() < ex[de]:
            # Aceita a nova configuração
            ener = ener+2*s[i]*h 
            mag  -= 2*s[i] 
            s[i] =- s[i]
    return ener,mag,s

@jit(nopython=True)
def metropolis(L, temp, s=None, steps=2000):
    """
    Simula uma rede LxL usando o algoritmo de Metropolis, à temperatura 'temp' (não é o beta)
    Pode receber o estado inicial 's'. Se não fornecido, é escolhido um estado aleatório inicial.
    Retorna três arrays: energia e magnetização ao longo dos passos do algoritmo e também o estado final
    Por default, são 2000 passos, mas pode ser fornecido outro valor
    """
    # Criação da rede e seus parâmetros
    N = L**2
    beta = 1/temp
    if s is None: s = random_energy_state(N)
    viz = neighbours(N, L)
    ener = energy_ising(s, viz)
    mag = np.abs(np.sum(s))
    
    # Loop principal
    Es, mags = [], []
    Es.append(ener)
    mags.append(mag)
    for i in range(steps):
        ener, mag, s = mcstep(beta, s, viz, ener, mag)
        Es.append(ener)
        mags.append(np.abs(mag))
    Es.pop()
    mags.pop()
    return Es, mags, s