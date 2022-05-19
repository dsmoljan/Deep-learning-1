# sve metode potrebne za rad CNN-a iz 2. zadatka

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import pdb
from utils import eval_perf_binary
from data_utils.data_utils import pad_collate_fn

SAVE_DIR = Path(__file__).parent / 'task3/'

class VanillaRNN(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
    
        self.embedding_matrix = embedding_matrix

        # 300 = input_size
        # 150 = number of params for one hidden state (cell)
        # 2 = number of RNN layers
        self.rnn = nn.RNN(300, 150, 2)
        self.fc1 = nn.Linear(150, 150, bias = True)
        self.fc_logits = nn.Linear(150, 1, bias = True)

        # inicijalizacija parametara
        self.reset_parameters()
    
    # TODO - prilagodi tako da uključiš i RNN slojeve
    def reset_parameters(self):
        for m in self.modules():
#             if isinstance(m, nn.RNN):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
#                 nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            self.fc_logits.reset_parameters()
        self.fc_logits.reset_parameters()
    
    # x je u obliku vektora intova
    # dakle npr. [32, 17, 1, 288, 3876, 1, 8976] -> njegova duljina je parametar T
    # i prije puštanja kroz mrežu treba ga pretvoriti 
    # u vektor korištenjem self.embedding_matrix
    # TODO - prilagodi RNN slojevima
    def forward(self, x_ind):
        
        # pretvori x u vektor floatova koristeći embedding matricu
        # treba paziti da nam sad x mora biti u obliku [T, batch, sequence_length], a NE [batch, T, sequence_length] što je default oblik x_vec
        x_vec = self.embedding_matrix(x_ind)#
        #T = x_vec.shape[1]
        
        # nakon ovog transponiranja oblik je [T, batch, sequence_length]
        x_vec = torch.transpose(x_vec, 0, 1)
        
        # ako ne damo inicijalno stanje za skrivene slojeve, jednostavno se koristi 0
        last_layer_features, hidden = self.rnn(x_vec)
        # last_layer_features - sadrži izlazne feature iz zadnjeg sloja RNN-a za svaki t
        # hidden = tenzor sa zadnjim skrivenim stanjem svakog elementa iz batcha
        
        # po uputama iz vježbe, kao ulaz u FC sloj dajemo  skriveno stanje iz zadnjeg sloja u zadnjem vremenskom koraku
        s1 = last_layer_features[-1]
        
        s2 = self.fc1(s1)
        h2 = torch.relu(s2)

        logits = self.fc_logits(h2)
        logits = torch.squeeze(logits)

        return logits
    
    # TODO - prilagodi, dodaj i RNN slojeve
    def get_params(self):
        param_list = []
        
        param_list.extend(self.rnn.parameters())
        param_list.extend(self.fc1.parameters())
        param_list.extend(self.fc_logits.parameters())
        
        # parametri embedding matrice, mozda...?
        
        return param_list
    
    # ovo bi trebalo biti isto
    def train(self, train_dataset, val_dataset, no_epochs = 10, lr = 1e-2, weight_decay = 0, batch_size = 1):
        self.cuda()
        
        # ovaj gubitak nam u sebi računa sigmoidu nad logitima
        # pa to ne moramo raditi prije
        bce_loss = nn.BCEWithLogitsLoss()
        
        dataloader_train = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=pad_collate_fn, drop_last=True)
        
        dataloader_val = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=pad_collate_fn, drop_last=True)
        
        optimizer = torch.optim.Adam(self.get_params(),lr=lr, weight_decay = weight_decay)

        loss_list = []
        loss_epoch = []
        
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
            for batch_num, (x_ind, y, lengths) in enumerate(dataloader_train):
                x_ind = x_ind.to(device="cuda")
                y = y.to(device="cuda")

                logits = self.forward(x_ind)
                loss = bce_loss(logits, y.float())
                loss_epoch.append(float(loss))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.get_params(), 0.25)
                optimizer.step()

                optimizer.zero_grad()

            # na kraju svake epohe provodimo validaciju i određujemo
            # early stopping
            mean_loss = np.mean(loss_epoch)
            print("Epoch:",epoch, ", loss:", mean_loss)
            loss_list.append(mean_loss)
            loss_epoch = []
            
            eval_loss, eval_avg_accuracy, eval_avg_precision, eval_avg_recall, eval_avg_f1, conf_matrix_eval = self.evaluate(dataloader_val, 32)
            train_loss, train_avg_accuracy, train_avg_precision, train_avg_recall, train_avg_f1, conf_matrix_train = self.evaluate(dataloader_train, 32)
            print('Eval set: Avg. loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f} F1: {:.4f}\n'.format(eval_loss, eval_avg_accuracy, eval_avg_precision, eval_avg_recall, eval_avg_f1))
            print("Conf matrix eval:", conf_matrix_eval)

            # early stopping
            if (eval_avg_accuracy > best_acc):
                best_acc = eval_avg_accuracy
                best_params = self.state_dict()
                best_epoch = epoch
            
                plot_data['train_loss'] += [mean_loss]
                plot_data['valid_loss'] += [eval_loss]
                plot_data['train_acc'] += [train_avg_accuracy]
                plot_data['valid_acc'] += [eval_avg_accuracy]
                #plot_data['lr'] += [lr_scheduler.get_last_lr()]
            
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print("Best epoch: ", best_epoch)
        self.load_state_dict(best_params)
        #plot_training_progress(SAVE_DIR, plot_data)
         
    def evaluate(self, dataloader_val, batch_size = 1):
        #self.eval()
        bce_loss = nn.BCEWithLogitsLoss()
        activation = nn.Sigmoid()
        loss_list = []
        accuracy_list = []
        recall_list = []
        precision_list = []
        f1_list = []
        conf_matrix = [0,0,0,0] 
        with torch.no_grad():
            for batch_num, (x_ind, y_true, lengths) in enumerate(dataloader_val):
                x_ind = x_ind.to(device="cuda")
                y_true = y_true.to(device="cuda")
                logits = self.forward(x_ind)
                loss = bce_loss(logits, y_true.float())
                loss_list.append(float(loss))
                y_out = activation(logits).round().int()
                
                y_true = y_true.clone().detach().cpu().numpy()
                y_out = y_out.clone().detach().cpu().numpy()

                accuracy, recall, precision, f1, conf_matrix_epoch = eval_perf_binary(y_out, y_true)
                accuracy_list.append(accuracy)
                recall_list.append(recall)
                precision_list.append(precision)
                f1_list.append(f1)
                for i in range(len(conf_matrix)):
                    conf_matrix[i] += conf_matrix_epoch[i]
        loss = np.mean(loss_list)
        accuracy = np.mean(accuracy_list)
        precision = np.mean(precision)
        recall = np.mean(recall_list)
        f1 = np.mean(f1_list)
        
        return loss, accuracy, precision, recall, f1, conf_matrix