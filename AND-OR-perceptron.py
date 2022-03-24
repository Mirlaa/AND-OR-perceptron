# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:31:36 2017

@author: mirla
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# AND
entradas = np.array([[0,0],[0,1], [1,0], [1,1]])
saidas = np.array([0,0,0,1])
# OR
#entradas = np.array([[0,0],[0,1], [1,0], [1,1]])
#saidas = np.array([0,1,1,1])

d = np.zeros([4])
pesos = np.array([0.0, 0.0])
taxaAprendizagem = 0.1
b=0

iteracoes = 10

erros = np.zeros([4])
erros_iter = np.zeros([iteracoes,4]) #10x4

#%%
# função de ativação: step function
def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

#Função para gerar a decision boundary
def gerar_espaco(pesos,b):
    pixels = 100
    eixo_x = np.arange(-0.5, 1.8, (1.8 + 0.5)/pixels)
    eixo_y = np.arange(-0.5, 1.8, (1.8 + 0.5)/pixels)
    xx, yy = np.meshgrid(eixo_x, eixo_y)
    pontos = np.c_[xx.ravel(), yy.ravel()]
    
    Z = np.zeros([pontos.shape[0]])
    for i in range(Z.shape[0]):
        U = pontos[i].dot(pesos)
        Z[i] = stepFunction(U+b)
    Z = Z.reshape(xx.shape)
    return xx,yy,Z

# função de subplots
fig = plt.figure(figsize=(15, 5))

def sub_plots(d,pesos,n,b):
    xx, yy, Z = gerar_espaco(pesos,b)
    plt.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=[0,0,1,1], y=[0,1,0,1], hue=d, s=90)    
    plt.xlim(-0.15,1.2)
    plt.ylim(-0.15,1.2)
    plt.title(str(n+1) + '° Iteração ')
    
#%%
# treinamento
for it in range(iteracoes):
    for i in range(entradas.shape[0]):
        U = entradas[i].dot(pesos)
        d[i] = stepFunction(U+b)
        erros[i] = saidas[i] - d[i]
        for j in range(pesos.shape[0]):
            pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erros[i])
        b += taxaAprendizagem*erros[i]
    fig.add_subplot(2,5,it+1)
    sub_plots(d,pesos,it,b)
    plt.tight_layout()
    ##salvar os erros por iteração
    erros_iter[it] = erros
plt.show()
#%%
# Plot de erros por iteração

fig = plt.figure(figsize=(15, 3))
for i in range(iteracoes):
    fig.add_subplot(2,5,i+1)
    plt.plot(['[0,0]','[0,1]','[1,0]','[1,1]'],erros_iter[i])
    plt.title(str(i+1) + '° Iteração ')
    plt.tight_layout()
plt.show()           
#%%
aaa = entradas[3].dot(pesos)