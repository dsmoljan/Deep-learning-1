import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from util import draw_conv_filters_pytorch, plot_training_progress
import pdb
from sklearn.metrics import confusion_matrix

SAVE_DIR = Path(__file__).parent / 'task4/'

 
# razlika izmedu torch.nn.Conv2D i torch.nn.functional.conv2d https://stackoverflow.com/questions/61018705/torch-nn-conv2d-does-not-give-the-same-result-as-torch-nn-functional-conv2d
    
class CNNCifar(nn.Module):
  # CIFAR-10 je RGB, pa je ulaz u mrežu trokanalan
  def __init__(self, in_channels = 3, class_count = 10):
    super(CNNCifar, self).__init__()
    
    # konvolucijski i max pool slojevi
    # conv 16,5
    self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 16, kernel_size=5, stride=1, padding=2, bias=True)
    # pool 3,2
    self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
    # conv 32,5
    self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride=1, padding=2, bias=True)
    # pool 3,2
    self.pool2 = nn.MaxPool2d(kernel_size=3, stride = 2)
    
    # potpuno povezani slojevi
    # ulaz fc1 je poravnan tenzor iz pool2, poravnavanje radimo u forward metodi
    self.fc1 = nn.Linear(32*7*7, 256, bias=True)
    self.fc2 = nn.Linear(256, 128, bias=True)
    self.fc_logits = nn.Linear(128, class_count, bias=True)

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
    #pdb.set_trace()
    
    # flattening the tensor to 1x256
    # počinjemo flattenanje od druge dimenzije, jer su 0-dimenzija samo indeksi podataka
    h2_flat = h2.flatten(start_dim = 1)
    
    # fc1
    s3 = self.fc1(h2_flat)
    h3 = torch.relu(s3)
    
    s4 = self.fc2(h3)
    h4 = torch.relu(s4)
    
    logits = self.fc_logits(h4)
    
    return logits

  # moja ideja je da ovdje proslijedis dataset objekt, i onda iz njega instanciraš dataloader
  # onda možeš u notebooku ovako https://nextjournal.com/gkoehler/pytorch-mnist instancirati MNIST dataset i samo ga proslijediti metodi
  # ovaj link općenito ima super primjer treniranja modela u PyTorchu
  # kada ćeš pripremati MNIST dataset, ne zaboravi ga poslati na GPU ovako https://stackoverflow.com/questions/65327247/load-pytorch-dataloader-into-gpu
  def train(self, train_x, train_y, val_x, val_y, weight_decay = 0, no_epochs = 4, batch_size = 50, lr = 0.1):
        # prebacujemo model na GPU
        self.cuda()
        
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        # ovdje još treba dodati LR scheduler
        
        CE_loss = nn.CrossEntropyLoss()
        # ne koristim softmax2d jer izlaz na kojem primjenjujem softmax je 1D tenzor
        activation = nn.Softmax(dim = 1)
                
        loss_list = []
        loss_epoch = []
        
        # generira indekse shufflanog polja
        ind_shuffled = torch.randperm(len(train_x))
        
        # shuffla polje
        train_x_shuffled = train_x[ind_shuffled, :] 
        train_y_shuffled = train_y[ind_shuffled]

        x_train_batches = [x for x in torch.split(train_x_shuffled, batch_size)]
        y_train_batches = [y for y in torch.split(train_y_shuffled, batch_size)]

        best_params = self.state_dict()
        best_acc = 0
        best_epoch = 0 # epoch early stop modela

        plot_data = {}
        plot_data['train_loss'] = []
        plot_data['valid_loss'] = []
        plot_data['train_acc'] = []
        plot_data['valid_acc'] = []
        plot_data['lr'] = []

        for epoch in range(no_epochs):
            for batch in range(len(x_train_batches)):
                
                x_minibatch = x_train_batches[batch].to(device="cuda")
                y_minibatch = y_train_batches[batch].to(device="cuda")
                
                # CE loss u sebi ima softmax, zato NE raditi softmax prije puštanja u loss!
                # https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
                # ali naravno, kad želiš dobiti izlaz iz mreže, onda koristi softmax nad logitsima
                loss = CE_loss(self.forward(x_minibatch), y_minibatch)
                loss_epoch.append(float(loss))
                
                loss.backward()
                
                optimizer.step()
                
                optimizer.zero_grad()
                
                #print("Epoch: (%3d) (%5d/%5d) | Crossentropy Loss:%.2e" %(epoch, batch + 1, len(x_train_batches), loss.item()))
            
            # ovo mozes zamijenit tensorboardom ako budes htio
            if (epoch == (no_epochs - 1)):
              draw_conv_filters_pytorch(self.conv1.weight.detach().cpu().numpy(), weight_decay,SAVE_DIR)
            mean_loss = np.mean(loss_epoch)
            print("Epoch:",epoch, ", loss:", mean_loss)
            eval_loss, eval_avg_accuracy, eval_avg_precision, eval_avg_recall = self.eval(val_x, val_y)
            train_loss, train_avg_accuracy, train_avg_precision, train_avg_recall = self.eval(train_x, train_y)
            print('Eval set: Avg. loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f} \n'.format(eval_loss, eval_avg_accuracy, eval_avg_precision, eval_avg_recall))
            loss_list.append(mean_loss)
            loss_epoch = []

            # early stopping
            if (eval_avg_accuracy > best_acc):
              best_acc = eval_avg_accuracy
              best_params = self.state_dict()
              best_epoch = epoch
            
              plot_data['train_loss'] += [mean_loss]
              plot_data['valid_loss'] += [eval_loss]
              plot_data['train_acc'] += [train_avg_accuracy]
              plot_data['valid_acc'] += [eval_avg_accuracy]
              plot_data['lr'] += [lr_scheduler.get_last_lr()]

            lr_scheduler.step()
        
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("Best epoch: ", best_epoch)
        self.load_state_dict(best_params)
        plot_training_progress(SAVE_DIR, plot_data)
   
  def predict(self, img):
        return torch.argmax(self.forward(img))
    
  def eval(self, eval_x, eval_y, batch_size = 1):
    correct = 0
    CE_loss = nn.CrossEntropyLoss()
    activation = nn.Softmax(dim = 1)
    with torch.no_grad():
        eval_x = eval_x.to(device='cuda')
        eval_y = eval_y.to(device='cuda')
        y_out = self.forward(eval_x)
        eval_loss= float(CE_loss(y_out, eval_y))
        y_pred = torch.argmax(activation(y_out), dim = 1)
        eval_y = eval_y.clone().detach().cpu().numpy()
        y_pred = y_pred.clone().detach().cpu().numpy()
        #pdb.set_trace()
        cm = confusion_matrix(eval_y, y_pred)
        #correct += y_pred.eq(y.data.view_as(y_pred)).sum()
        tp_and_fn = cm.sum(1)
        tp_and_fp = cm.sum(0)
        tp = cm.diagonal()
        avg_accuracy = sum(tp)/len(eval_y)
        avg_precision = np.mean(tp / tp_and_fp)
        avg_recall = np.mean(tp / tp_and_fn)
        #pdb.set_trace()
        # mislim da ti ova metoda izbaci tp, accuracy i recall po klasama https://stackoverflow.com/questions/40729875/calculate-precision-and-recall-in-a-confusion-matrixs
        # pa sad mozes odlucit oces vratit sammo to il ces izracunat prosjek
        return eval_loss, avg_accuracy, avg_precision, avg_recall
