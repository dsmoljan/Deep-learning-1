# zadatak - proširiti izvedbu logreg tako da
# omogućimo jednostavno zadavanje potpuno povezanih modela proizvoljne dubine
# znači korak dalje nego 2. zadatak, gdje smo mogli samo odrediti 
# broj neurona u jednom jedinom skrivenom sloju
# ovdje, bi (u teoriji) mogli zadati arhitekturu (2x32x64x128x64x32x8)
# ili bilo što slično
# nulti element liste - dimenzionalnost podataka
# zadnji element liste - broj neurona u izlaznom sloju (broj klasa)
# sve između - skriveni slojevi
# (2x3) -> logistička regresija 2D podataka u 3 razreda

import torch
from torch import nn
import torch.optim as optim
from torch.linalg import norm
import numpy as np
import pdb
from data import eval_perf_multi

class PTDeep(nn.Module):
  def __init__(self, arch, activation = torch.relu, cuda = False):
    super().__init__()
    """Arguments:
       - arch: architecture of the FNN
    """
    
    weights = []
    biases = []
    
    self.arch = arch
    self.activation = activation
    
    for i in range(len(arch) - 1):
        weight_i = torch.tensor(np.zeros((arch[i], arch[i + 1])),dtype=torch.float,device="cuda" if cuda else None)

        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/ --> He (Kaming) inicijalizacija najbolja za ReLU 
        weights.append(nn.Parameter(nn.init.kaiming_uniform_(weight_i, mode='fan_in', nonlinearity='relu'), requires_grad=True))
        bias_i = nn.Parameter(torch.zeros(arch[i + 1],device="cuda" if cuda else None), requires_grad=True)
        biases.append(bias_i)
        
        
#         weight_i = torch.nn.Parameter(torch.rand(arch[i], arch[i+1]), requires_grad=True)
#         weights.append(weight_i)
#         bias_i = torch.nn.Parameter(torch.zeros(1,arch[i+1]), requires_grad=True)
#         biases.append(bias_i)
    
    self.weights = torch.nn.ParameterList(weights)
    self.biases = torch.nn.ParameterList(biases)
    
    # preostali su nam X, koji ćemo dobiti kao ulaz modela, h1...hn, koje računamo pomoću W i b 
    # u metodi forward, te y, koji nam je izlaz modela i koji je konačni rezultat metode forward

  def forward(self, X):
    # unaprijedni prolaz modela: izračunati vjerojatnosti
    # koristiti: torch.mm, torch.softmax
    # znači ovdje definiramo Y = softmax(X*w + b)
    # torch.mm = matrično množenje, kao np.dot
    
    h_prev = X
    for i in range(len(self.weights) - 1):
        h_cur = torch.mm(h_prev,self.weights[i]) + self.biases[i]
        h_cur = self.activation(h_cur)
        h_prev = h_cur
    
    h_final = torch.mm(h_prev,self.weights[len(self.weights) - 1]) + self.biases[len(self.biases) - 1]
    probs = torch.softmax(h_final, dim = 1)
    
    return probs
    
  # TODO: mozda i ne treba mijenjati
  def get_loss(self, X, Yoh_):
    h = self.forward(X)
    
    loss = -1*torch.mean(torch.sum(torch.log(h)*Yoh_, dim = 1))
    return loss

# ako stavimo param_lambda = 0, de-facto smo isključili regularizaciju
# primijeti da je ovo doslovno kopirana implementacija iz pt_logreg 
# korištenjem OO Strategtija, gdje kao model predajemo "sučelje"
# postigli smo da nam petlja za treniranje ne ovisi o korištenom modelu
def train(model, X, Yoh_, param_niter, param_delta, param_lambda = 0,debug = True):  
    # inicijalizacija optimizatora, momentum stavljamo na 0.9 https://12ft.io/proxy?q=https%3A%2F%2Ftowardsdatascience.com%2Fwhy-0-9-towards-better-momentum-strategies-in-deep-learning-827408503650
    optimizer = optim.SGD(params=model.parameters(), lr=param_delta, momentum=0.9, weight_decay = param_lambda) 

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    loss = 0
    losses = []
    for i in range(param_niter):
        # dovoljno je da funkciji za loss pošaljemo X i Yoh_, jer ona pristupa 
        # modelu i pomoću njega računa Y_out
        loss = model.get_loss(X, Yoh_)

        # računanje gradijenata
        loss.backward()

        # korak optimizacije
        optimizer.step()
        
        if (debug and i % 10 == 0):
            print(f'step: {i}, loss:{loss}')
        
        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()
        losses.append(float(loss.clone().detach().cpu().numpy()))

    
    print("Final loss: ", loss.clone().detach().cpu().numpy())
    return losses

# nikad ne pokretat ovo sa mb_size = 1, trajat će vječno
def train_mb(model, X, Yoh_, param_niter, param_delta, param_lambda = 0, mb_size = 1, debug = False):
    optimizer = optim.SGD(params=model.parameters(), lr=param_delta, momentum=0.9, weight_decay = param_lambda) 

    loss = 0
    losses = []
    for i in range(param_niter):
        #https://discuss.pytorch.org/t/shuffle-images-ina-tensor-during-training/2157

        ind_shuffled = torch.randperm(len(X)) # generira indekse shufflanog polja
        shuffled_X = X[ind_shuffled, :] # shuffla polje
        shuffled_Y = Yoh_[ind_shuffled, :]

#         tensor_1 = X[1, :] # ovo je tako dobro, mozes ovako indeksirati određene dimenzije tenzora
#         tensor_2 = X[2, :]
#         tensor_r = torch.vstack([tensor_1, tensor_2])
#         tensor_r.shape

        x_train_batches = [x for x in torch.split(shuffled_X, mb_size)]
        y_train_batches = [y for y in torch.split(shuffled_Y, mb_size)]
        
        for b in range(len(x_train_batches)):
            x_minibatch = x_train_batches[b]
            y_minibatch = y_train_batches[b]

            loss = model.get_loss(x_minibatch, y_minibatch)

            # računanje gradijenata
            loss.backward()

            # korak optimizacije
            optimizer.step()

            if (debug and i % 10 == 0):
                print(f'step: {i}, loss:{loss}')

            # Postavljanje gradijenata na nulu
            optimizer.zero_grad()
            
        losses.append(float(loss.clone().detach().cpu().numpy()))

    
    print("Final loss: ", loss.clone().detach().cpu().numpy())
    return losses


def train_early_stop(model, X, Yoh_,val_x,val_y, param_niter, param_delta, param_lambda = 0,debug = False):
    optimizer = optim.SGD(params=model.parameters(), lr=param_delta, momentum=0.9, weight_decay = param_lambda) 

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    loss = 0
    losses = []
    best_weights = model.weights
    best_biases = model.biases
    best_acc = 0
    best_epoch = 0 # epoch early stop modela
                     
    for i in range(param_niter):
        # dovoljno je da funkciji za loss pošaljemo X i Yoh_, jer ona pristupa 
        # modelu i pomoću njega računa Y_out
        loss = model.get_loss(X, Yoh_)

        # računanje gradijenata
        loss.backward()

        # korak optimizacije
        optimizer.step()
        
        if (debug and i % 10 == 0):
            print(f'step: {i}, loss:{loss}')
        
        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()
        losses.append(float(loss.clone().detach().cpu().numpy()))
                     
        # provjera za early stopping
                     
        y_pred_val = eval(model, val_x)
        accuracy, pr, avg_precision, avg_recall = eval_perf_multi(val_y, y_pred_val)
        if (accuracy > best_acc):
            best_weights = model.weights
            best_biases = model.biases
            best_acc = accuracy
            best_epoch = i

    
    # na kraju vraćamo model sa najboljim accuracyem
    model.weights = best_weights
    model.biases = best_biases
    print("Early stop at epoch", best_epoch)
    print("Final loss: ", loss.clone().detach().cpu().numpy())
    return losses

def train_adam(model, X, Yoh_, param_niter, param_delta, param_lambda = 0, variable_lr = False, debug = True):  
    # inicijalizacija optimizatora, momentum stavljamo na 0.9 https://12ft.io/proxy?q=https%3A%2F%2Ftowardsdatascience.com%2Fwhy-0-9-towards-better-momentum-strategies-in-deep-learning-827408503650
    optimizer = optim.Adam(params=model.parameters(), lr=param_delta, weight_decay = param_lambda) 
    if (variable_lr):
      scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    loss = 0
    losses = []
    for i in range(param_niter):
        # dovoljno je da funkciji za loss pošaljemo X i Yoh_, jer ona pristupa 
        # modelu i pomoću njega računa Y_out
        loss = model.get_loss(X, Yoh_)

        # računanje gradijenata
        loss.backward()

        # korak optimizacije
        optimizer.step()

        # pozivamo nakon svake epohe
        if (variable_lr):
          scheduler.step()
        
        if (debug and i % 10 == 0):
            print(f'step: {i}, loss:{loss}')
        
        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()
        losses.append(float(loss.clone().detach().cpu().numpy()))


    
    print("Final loss: ", loss.clone().detach().cpu().numpy())
    return losses

# također kopirano iz pt_logreg, provjeri jel oke
def eval(model, X):
  """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
  """
  X_tensor = torch.tensor(X, dtype=torch.float)
  Y_pred = model.forward(X).detach().cpu().numpy()
  return np.argmax(Y_pred, axis = 1)

# ispisuje simboličko ime i dimenzije tenzora svih parametara
# također, računa ukupan broj parametara modela
def count_params(model):
    # rezultatm poziva model.parameters je lista tupleova
    # 0 član tuplea je ime tenzora, a 1 član je sam tenzor
    no_params = 0
    for param in list(model.named_parameters()):
        name = param[0]
        tensor = param[1]
        shape = list(tensor.shape)
        if (len(shape) == 2):
            no_params += shape[0] * shape[1]
        else:
            no_params += shape[0] # za bias, kojem je shape skalar
        print(f'Name: {name}, shape:{shape}')
    print("No params:", no_params)
   # print(list(model.named_parameters())) 