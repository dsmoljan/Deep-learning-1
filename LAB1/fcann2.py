import numpy as np
from data import class_to_onehot
import pdb

#model ima 1 skriveni sloj -> parametri su W1, W2, b1, b2
# kao aktivaciju nakon 1. sloja koristimo ReLU, a kao aktivaciju 
# zadnjeg (izlaznog) sloja koristimo softmax
# IZUZETNO BITNO! Ako hoćeš modelirati iole složenije distribucije, nek ti skriveni sloj (layer_size) ima barem 8 neurona
# param_delta nek ti onda bude 0.001, a param_niter 5000-10000
def fcann2_train(X, Y, param_niter=10000, param_delta=0.001, debug = False):
    # ovdje isto koristimo one_hot matricu (oznaka Y')ž
    
    # veličina skrivenog sloja - 3 je u primjeru za labos
    layer_size = 8
    
    input_dim = 2 # dimezija ulaznih podataka, uvijek 2 - npr. (0.25, -0.35)
    no_C = np.max(Y) + 1 # alternativno, ovdje bi mogli uzeti y_one_hot.shape[1], 
                                # tj. broj stupaca matrice one_hot jer on odgovara broju različitih klasa
        
    W1 = np.random.randn(input_dim, layer_size)
    W2 = np.random.randn(layer_size, no_C)
    b1 = np.zeros(layer_size)
    b2 = np.zeros(no_C)
    
    y_one_hot = class_to_onehot(Y)
    
    loss = 0
    
    for i in range(param_niter):
        # forward pass - dobivamo izlaze mreže
        s1, s2, y_out = fcann2_forward(X, W1, W2, b1, b2)
        
        # trenutno ti je glavni problem taj što ti y_out nije dobar - u svakom redu bi zbroj trebao biti 1, a tebi to nije
        
        # računanje gubitka
        loss = -1*np.mean(np.sum(np.log(y_out)*y_one_hot, axis = 1))
        
        #pdb.set_trace()
        
        if (debug and i % 10 == 0):
            print(f'step: {i}, loss:{loss}')
        
        # računanje gradijenata - cilj nam je dobiti grad_W1, grad_b1 itd
        
        # gradijenti gubitka s obzirom na linearnu mjeru drugog sloja u svim podacima (s2)
        # y_out = probs
        #dL_ds2 = y_out - y_one_hot # N x C
        dL_ds2 = y_out - y_one_hot # N x C

        
        # gradijenti gubitka s obzirom na parametre drugog sloja - ovo bi trebalo biti ok
        h1 = ReLU(s1)
        grad_W2 = np.dot(h1.transpose(),dL_ds2) # C x H
        grad_b2 = np.sum(y_out - y_one_hot, axis = 0)
        
        # gradijent gubitka s obzirom na nelinearni izlaz prvog sloja u svim podacma
        # Jakobijan linearnog sloja -> matrica težina W2
        # Jakobijan zglobnice -> dijagonalna matrica koja na dijagonali ima 0 i 1, ovisno o predznaku odgovarajuće komponente prvog sloja
        dL_ds1 = np.dot((y_out - y_one_hot),W2.transpose())
        dL_ds1[s1 <= 0] = 0 # gradijent prolazi samo tamo gdje je aktivacija bila veća od 0, na ostalim mjestima je 0
        
        grad_W1 = np.dot(X.transpose(),dL_ds1)
        grad_b1 = np.sum(dL_ds1, axis = 0)
        
        W1 += -param_delta * grad_W1
        W2 += -param_delta * grad_W2
        b1 += -param_delta * grad_b1
        b2 += -param_delta * grad_b2
    
    print("Final loss: ", loss)
    return W1, W2, b1, b2


# provodi jedan unaprijedni korak fcann2 mreže
def fcann2_forward(X, W1, W2, b1, b2):
    s1 = np.dot(X, W1) + b1
    h1 = ReLU(s1)
    s2 = np.dot(h1, W2) + b2
    # out = P(Y|x), za dvije klase to će biti vektor [p0 p1], koji
    # nam daje vjeorjatnosti da ulazni primjer pripada c0, tj c1
    out = stable_softmax(s2)
    return s1,s2,out # out === probs 

def fcann2_classify(X, W1, W2, b1, b2):
  Y_pred = fcann2_forward(X, W1, W2, b1, b2)[2]
  return np.argmax(Y_pred, axis = 1)

def ReLU(x):
    return np.maximum(0, x)

# stabilni softmax - poglavlje 4.1 Deeplearningbook
def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted, axis = 1)[:, np.newaxis]
    return probs

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))
