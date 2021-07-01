import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import argparse
import pickle

@jit(nopython=True)
def neighbours(N, L):
    """
    Constrói a matriz de vizinhança.
    Assume que há uma "vizinhança circular" como se a rede 
    fosse um toro.
    Adaptação do algoritmo proposto no enunciado
    """
    # Cria uma matriz Nx4
    viz = np.zeros((N,4), dtype=np.longlong)

    # (k,0): vizinho à direita
    # (k,1): vizinho acima
    # (k,2): vizinho à esquerda
    # (k,3): vizinho abaixo]
    for k in range(N):
        # Vizinho à direita (k,0)
        viz[k][0] = k+1
        if (k % L == L-1):
            # Sítio na borda da direita
            viz[k][0] = k+1-L

        # Vizinho acima (k,1)
        viz[k][1] = k+L
        if (k >= N-L):
            # Sítio na borda superior
            viz[k][1] = k+L-N

        # Vizinho à esquerda (k,2)
        viz[k][2] = k-1
        if (k % L == 0):
            # Sítio na borda da esquerda
            viz[k][2] = k+L-1

        # Vizinho abaixo (k,3)
        viz[k][3] = k-L
        if (k < L):
            # Sítio na borda inferior
            viz[k][3] = k+N-L
    return viz

@jit(nopython=True)
def energy_ising(s, viz):
    """
    Calcula a energia da rede Ising usando os vizinhos de cada sítio.
    É importante notar que precisamos apenas dos 
    vizinhos à direita e acima
    """
    E = 0
    N = len(s)
    for i in range(N):
        right = viz[i][0]
        up = viz[i][1]
        h = s[right] + s[up]
        E = E - s[i]*h
    return E

@jit(nopython=True)
def sub_min(H):
    """
    Retorna o mínimo do vetor H, ignorando o segundo e penúltimo itens
    """
    n = len(H)
    menor = H[0]
    for i in range(n):
        if (i == 1 or i == n-2):
            continue
        if (H[i] < menor):
            menor = H[i]
    return menor

@jit(nopython=True)
def energy_scale(N,E):
    """
    Escala os valores de energia para que fiquem entre 0 e N 
    ao invés de -2N até 2N
    """
    E = E+2*N
    E = E/4
    if (int(E) - E > 0):
        print("Energia não inteira pós escala!")
    return int(E)

@jit(nopython=True)
def random_energy_state(N):
    """
    Gera um estado aleatório de energia
    """
    s = []
    # De 0 a N-1
    for i in range(N):
        s.append(np.sign(2*np.random.random()-1))
    s = np.array(s)
    return s

@jit(nopython=True)
def wang_landau(N, E, s):
    """
    Algoritmo de Wang-Landau para estimativa da densidade de estados g(E)
    """
    # Inicialização arrays, de 0 a N, inclusive
    lnG = np.zeros(N+1, dtype=np.float64)
    H = np.zeros(N+1, dtype=np.int64)
    Hc = np.zeros(N+1, dtype=np.int64)
    mmicro = np.zeros(N+1, dtype=np.float64)
    viz = neighbours(N, np.sqrt(N))
    
    # Variáveis do Loop Principal
    lnf = 1.0
    flat = False
    m = s.sum()
    for i in range(10**7):
        for j in range(N):
            # Escolhe um sítio aleatório
#             rng = np.random.default_rng()
            k = np.random.randint(N)
            
            # Soma sobre os vizinhos de k
            h = 0
            for v_idx in viz[k]:
                h += s[v_idx]
            
            # Energia desse novo estado
            E2 = E + s[k]*h/2
            if (int(E2) - E2 > 0): print("Energia não inteira pós iteração!")
            E2 = int(E2)

            # Economizar algumas exponenciais
            if lnG[E] > lnG[E2]:
                # Faz a troca
                s[k] = -s[k]
                E = E2
                m = m - 2*s[k]
            else:
#                 rng = np.random.default_rng()
                P = np.exp(lnG[E] - lnG[E2])
                if (np.random.random() < P):
                    # Faz a troca
                    s[k] = -s[k]
                    E = E2
                    m = m - 2*s[k]
            H[E] = H[E] + 1
            lnG[E] = lnG[E] + lnf
            ## Usando o abs no mmicro (?)
            mmicro[E] = mmicro[E] + m
            
        if (i % 1000 == 0):
            hmed = np.sum(H)/(N-1)
            hmin = sub_min(H)

            if (hmin > 0.8*hmed):
                Hc = H.copy()
                H = np.zeros(N+1, dtype=np.int64)
                lnf = lnf/2
                if (i % 10000 == 0):
                    print(round(hmed,2),round(hmin,2),round(hmin,2),
                        round(0.8*hmed,2), i, lnf)
        if (lnf < 10**-8):
            break
    mmicro = mmicro/Hc
    lnG0 = lnG[0]
    lnG = lnG - lnG0 + np.log(2)
    return lnG, mmicro

def main():
    # Parse dos argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("L", help="Tamanho da rede LxL")
    args = parser.parse_args()
    L = int(args.L)

    N = L**2
    s = random_energy_state(N)
    viz = neighbours(N,np.sqrt(N))
    E = energy_ising(s,viz)
    E = energy_scale(N,E)
    lnG, mmicro = wang_landau(N, E, s)
    with open("wang"+str(L)+"x"+str(L)+".dict", "wb") as f:
        pickle.dump(lnG, f)
    with open(f"mag"+str(L)+"x"+str(L)+".dict", "wb") as f:
        pickle.dump(mmicro, f)
    print("Criado rede "+str(L)+"x"+str(L))

if __name__=="__main__":
    main()