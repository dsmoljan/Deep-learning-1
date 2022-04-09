import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from util import draw_conv_filters_pytorch
import pdb

SAVE_DIR = Path(__file__).parent / 'task3/'

 
# razlika izmedu torch.nn.Conv2D i torch.nn.functional.conv2d https://stackoverflow.com/questions/61018705/torch-nn-conv2d-does-not-give-the-same-result-as-torch-nn-functional-conv2d
    
class CNN(nn.Module):
  def __init__(self, in_channels = 1, class_count = 10):
    super().__init__()
    
    # konvolucijski i max pool slojevi
    self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 16, kernel_size=5, stride=1, padding=2, bias=True)
    self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride=1, padding=2)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride = 2)
    
    # potpuno povezani slojevi
    # ulaz fc1 je poravnan tenzor iz pool2, poravnavanje radimo u forward metodi
    self.fc1 = nn.Linear(32*7*7, 512, bias=True)
    self.fc_logits = nn.Linear(512, class_count, bias=True)

    # parametri su već inicijalizirani pozivima Conv2d i Linear
    # ali možemo ih drugačije inicijalizirati
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    s1 = self.conv1(x)
    s1 = self.pool1(s1)
    h1 = torch.relu(s1)  # može i h.relu() ili nn.functional.relu(h)
    
    s2 = self.conv2(h1)
    s2 = self.pool2(s2)
    h2 = torch.relu(s2)
    
    # flattening the tensor to 1x512
    h2_flat = h2.view(h2.shape[0], -1)
    
    s3 = self.fc1(h2_flat)
    h3 = torch.relu(s3)
    logits = self.fc_logits(h3)
    
    return logits

  # moja ideja je da ovdje proslijedis dataset objekt, i onda iz njega instanciraš dataloader
  # onda možeš u notebooku ovako https://nextjournal.com/gkoehler/pytorch-mnist instancirati MNIST dataset i samo ga proslijediti metodi
  # ovaj link općenito ima super primjer treniranja modela u PyTorchu
  # kada ćeš pripremati MNIST dataset, ne zaboravi ga poslati na GPU ovako https://stackoverflow.com/questions/65327247/load-pytorch-dataloader-into-gpu
  def train(self, train_dataset, test_dataset, weight_decay = 0, no_epochs = 4, batch_size = 50, lr = 0.1):

        # prebacujemo model na GPU
        self.cuda()
        
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay = weight_decay)
        optimizer = torch.optim.SGD([
            {"params": [*self.conv1.parameters(),
                        *self.conv2.parameters(),
                        *self.fc1.parameters(),
                        *self.fc_logits.parameters()], "weight_decay": weight_decay}
        ], lr=lr)
        
        # super stvar kod pytorchovog CE lossa je da interno radi encoding y-labela u one hot
        # https://stackoverflow.com/questions/62456558/is-one-hot-encoding-required-for-using-pytorchs-cross-entropy-loss-function
        # tako da mu predajemo listu y-oznaka (npr. [1,5,3,4] itd) a on to pretvara u one hot
        CE_loss = nn.CrossEntropyLoss()
        # ne koristim softmax2d jer izlaz na kojem primjenjujem softmax je 1D tenzor
        activation = nn.Softmax(dim = 1)
        
        dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        loss_list = []
        loss_epoch = []

        
        for epoch in range(no_epochs):
            for i, (x, y) in enumerate(dataloader_train):
                x = x.to(device="cuda")
                y = y.to(device="cuda")
                
                # CE loss u sebi ima softmax, zato NE raditi softmax prije puštanja u loss!
                # https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
                # ali naravno, kad želiš dobiti izlaz iz mreže, onda koristi softmax nad logitsima
                loss = CE_loss(self.forward(x), y)
                loss_epoch.append(float(loss))
                
                loss.backward()
                
                optimizer.step()
                
                optimizer.zero_grad()
                
                #print("Epoch: (%3d) (%5d/%5d) | Crossentropy Loss:%.2e" %(epoch, i + 1, len(dataloader_train), loss.item()))
            
            # ovo mozes zamijenit tensorboardom ako budes htio
            if (epoch == (no_epochs - 1)):
              draw_conv_filters_pytorch(self.conv1.weight.detach().cpu().numpy(), weight_decay, SAVE_DIR)
            mean_loss = np.mean(loss_epoch)
            print("Epoch:",epoch, ", loss:", mean_loss)
            self.eval(test_dataset)
            loss_list.append(mean_loss)
            loss_epoch = []
        
        return loss_list
   
  def predict(self, img):
        return torch.argmax(self.forward(img))
    
  def eval(self, eval_dataset, batch_size = 1):
    dataloader_eval = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loss = []
    correct = 0
    CE_loss = nn.CrossEntropyLoss()
    activation = nn.Softmax(dim = 1)
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader_eval):
            x = x.to(device='cuda')
            y = y.to(device='cuda')
            y_out = self.forward(x)
            eval_loss.append(float(CE_loss(y_out, y)))
            y_pred = torch.argmax(activation(y_out))
            correct += y_pred.eq(y.data.view_as(y_pred)).sum()
        eval_loss = np.mean(eval_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(eval_loss, correct, len(eval_dataset),100. * correct /len(eval_dataset)))