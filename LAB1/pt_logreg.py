import torch
from torch import nn
import torch.optim as optim
from torch.linalg import norm
import numpy as np
import pdb


class PTLogreg(nn.Module):
  def __init__(self, D, C):
    super().__init__()
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
    """

    # inicijalizirati parametre (koristite nn.Parameter):
    # imena mogu biti self.W, self.b
    # zašto nn.Parameter - da bi se varijabla povezala sa moudlom PTLogreg, te 
    # da bi joj mogli pristupiti kroz module.parameters
    # https://stackoverflow.com/questions/50935345/understanding-torch-nn-parameter
    self.W = torch.nn.Parameter(torch.rand(D,C), requires_grad=True)
    self.b = torch.nn.Parameter(torch.zeros(C), requires_grad=True)

  def forward(self, X):
    # unaprijedni prolaz modela: izračunati vjerojatnosti
    # koristiti: torch.mm, torch.softmax
    # znači ovdje definiramo Y = softmax(X*w + b)
    # torch.mm = matrično množenje, kao np.dot
    scores = torch.mm(X, self.W) + self.b # N x C
    probs = torch.softmax(scores, dim = 1)
    return probs
    
    
  def get_loss(self, X, Yoh_):
    # formulacija gubitka
    # koristiti: torch.log, torch.mean, torch.sum
    # https://medium.com/deeplearningmadeeasy/negative-log-likelihood-6bd79b55d8b6 -> lijepo objašnjenje intuicije iza korištenja negativne log izglednosti 
    # kao funkcije gubitka
    h = self.forward(X)
    
    loss = -1*torch.mean(torch.sum(torch.log(h)*Yoh_, dim = 1))
    #pdb.set_trace()
    return loss
   

# ako stavimo param_lambda = 0, de-facto smo isključili regularizaciju
def train(model, X, Yoh_, param_niter, param_delta, param_lambda = 0,debug = True):  
    # inicijalizacija optimizatora
    optimizer = optim.SGD(params=model.parameters(), lr=param_delta)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    loss = 0
    for i in range(param_niter):
        # dovoljno je da funkciji za loss pošaljemo X i Yoh_, jer ona pristupa 
        # modelu i pomoću njega računa Y_out
        loss = model.get_loss(X, Yoh_) + param_lambda * norm(torch.flatten(model.W), ord=2)

        # računanje gradijenata
        loss.backward()

        # korak optimizacije
        optimizer.step()
        
        if (debug and i % 10 == 0):
            print(f'step: {i}, loss:{loss}')
        
        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()
    
    print("Final loss: ", loss.clone().detach().cpu().numpy())


def eval(model, X):
  """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
  """
  X_tensor = torch.tensor(X, dtype=torch.float)
  Y_pred = model.forward(X).detach().cpu().numpy()
  return np.argmax(Y_pred, axis = 1)
  # ulaz je potrebno pretvoriti u torch.Tensor
  # izlaze je potrebno pretvoriti u numpy.array
  # koristite torch.Tensor.detach() i torch.Tensor.numpy()