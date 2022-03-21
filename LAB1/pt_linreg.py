# cilj je odrediti parametre pravca y = a*x + b, koji prolazi
# kroz točke (1,3) i (2,5)
# dakle, kao ulaze dajemo X - [1,3], te pokušavamo naučiti NN da nauči tu funkciju
# datoteka pt_linreg.py -> PAZI, OVO MORA BITI SVAKO U SVOJOJ DATOTECI!

import torch
import torch.nn as nn
import torch.optim as optim

def linreg_train(X = torch.tensor([0, 1, 2]), Y = torch.tensor([1, 3, 5])):
    print("Ovdje")
    ## Definicija računskog grafa
    # podaci i parametri, inicijalizacija parametara -> ovo su parametri modela, kao w i b kod logističke
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    X = torch.tensor([0, 1, 2])
    Y = torch.tensor([1, 3, 5])

    # optimizacijski postupak: gradijentni spust (SGD)
    optimizer = optim.SGD([a, b], lr=0.1)

    for i in range(100):
        # afin regresijski model
        # Y = prave oznake, Y_ = oznake koje je predvidio model
        Y_ = a*X + b

        n = len(Y)

        # kvadratni gubitak - squared error (ne još MSE!)
        # kako bi postigli da iznosi gradijenata budu neovisni o broju
        # podataka, trebamo koristiti MSE da uprosječimo gubitak
        # sad, kad koristimo SE, a ne MSE, povećavanjem broja podataka
        # loss bi rastao, što znači da bi iznosi gradijenata bili ovisni
        # o broju podataka
        # znači, samo zamijeni ovu funkciju sa svojom implementacijom MSE
        loss = MSE(Y, Y_)
        
        # analitički gradijent
        grad_a_calc = (2/n)*torch.sum(X*(a*X + b - Y))
        grad_b_calc = (2/n)*torch.sum(a*X + b - Y)

        # računanje gradijenata
        loss.backward()

        # korak optimizacije
        optimizer.step()

        if (i % 10 == 0):
            print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}, a_grad: {a.grad}, b_grad: {b.grad} a_grad_calc: {grad_a_calc}, b_grad_calc: {grad_b_calc}\n')
        
        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()
    
    return a,b

def MSE(Y, Y_):
    diff = (Y-Y_)**2
    n = len(Y)
    return torch.sum(diff)/n
    

    
# za podzadatak 4, kad ti kaže "odredite analitičke izraze za izračun gradijenta"
# pa, to znači da samo pronađeš formulu pomoću koje se računa gradijent, npr. y' = 5x + 28 (bezveze pišem), to možeš ručno doslovno naći tu formulu
# i onda samo tu dobivenu formulu implementirati tako da ubaciš y' = 5x + 28 lol
    

