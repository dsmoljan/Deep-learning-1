# svaki objekt Vocabulary u sebi ima 2 rječnika, izgrađena nad istim skupom podataka
# itos - index to string
# stoi - string to index

#SPECIJALNI ZNAKOVI
# koristimo 2 specijalna znaka: 
# <PAD> - potreban kako bi naše batcheve, koji se sastoje od primjera različite duljine sveli na istu duljinu
# <UNK> - služi za riječi koje nisu u nađem vokabularu

import pdb
import torch
import torch.nn
import numpy as np

DIM = 300

class Vocabulary:
    # max_size = maksimalni broj riječi u rječniku
    # min_freq = minimalna frekvencija koju token mora imati da bi ga se spremilo u vokabular
    # posebni znakovi ne prolaze ovu provjeru
    
    # vokabular se gradi temeljem rječnika frekvencija za neko polje
    # rječnik frekvencija kao ključeve sadrži sve tokene koji su se pojavili u tom polju, vrijednosti = broj pojavljivanja
    def __init__(self, freq_dict = None, max_size = 1000, min_freq = 1, special_chars = ["<PAD>", "<UNK>"]):
        self.itos_dict = dict()
        self.stoi_dict = dict()
        counter = 0
        
        for elem in special_chars:
            self.stoi_dict.update({elem:counter})
            self.itos_dict.update({counter:elem})
            counter += 1
        
        sorted_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True))
        for token in sorted_dict.keys():
            word = token
            freq = int(sorted_dict[token])
            if ((max_size != -1 and counter >= max_size) or freq < min_freq):
                break
            self.stoi_dict.update({word:counter})
            self.itos_dict.update({counter:word})
            counter += 1
        
        #pdb.set_trace()
    
    # pretvara niz tokena (ili jedan token) u brojeve na osnovu svog rječnika stoi
    def encode(self, token_list):
        index_list = []
        for token in token_list:
            if token in self.stoi_dict:
                index_list.append(self.stoi_dict[token])
            else:
                index_list.append(self.stoi_dict["<UNK>"])
        return index_list
    
    # pretvara niz indeksa u tokene na osnovu svog rječnika itos
    def decode(self, index_list):
        token_list = []
        for index in index_list:
            if (index not in self.itos_dict):
                raise RuntimeException("Pogreška! Nepoznati indeks!")
            token_list.append(self.itos_dict[index])
        
        return token_list
    
    # na osnovu prednaučenih vektorskih reprezentacija zapisanih 
    # u datoteci path generira embedding matricu za vokabular
    # ako path nije zadan, koriste se slučajno inicijalizirane reprezentacije
    # vektorske reprezentacije koje koristimo u vježbi uvijek će biti 300-dimenzijske
    def get_embedding_matrix(self, path = None):
        V_size = len(self.stoi_dict)
        embedding_matrix = np.random.normal(0, 1, size = (V_size, DIM))
        embedding_matrix[0] = np.zeros(DIM)
        freeze = False
        if (path != None):
            freeze = True
            embeddings_dict = dict()
            # učitavamo datoteku s embeddinzima tokena i spremamo ih u rječnik
            with open(path, 'r') as file:
                while(True):
                    line = file.readline()
                    if not line:
                        break
                    ind = line.find(" ")
                    token = line[0:ind]
                    #print(token)

                    #pdb.set_trace()

                    x = np.array(['1.1', '2.2', '3.3'])
                    y = x.astype(np.float)

                    vector_reps = np.array(line[ind+1::].split(" ")).astype(np.float)
                    embeddings_dict.update({token:vector_reps})
        
            # na osnovu embedding rječnika i tokena iz rječnika kreiramo
            # embedding matricu
            i = 2
            #pdb.set_trace()
            for token in self.stoi_dict.keys():
                if (token == "<UNK>" or token == "<PAD>"):
                    #print("Skipping:",token)
                    continue
                if (token in embeddings_dict):
                    #print("Ovdje:", token)
                    embedding_matrix[i] = embeddings_dict[token]
                i += 1
        
        # vraćamo ju u optimiziranom omotaču
        embedding_matrix_tensor = torch.FloatTensor(embedding_matrix)
        return torch.nn.Embedding.from_pretrained(embedding_matrix_tensor, freeze = freeze, padding_idx = 0)
        
        def get_embedding_matrix(self, path = None):
        V_size = len(self.stoi_dict)
        embedding_matrix = np.random.normal(0, 1, size = (V_size, DIM))
        embedding_matrix[0] = np.zeros(DIM)
        freeze = False
        if (path != None):
            freeze = True
            embeddings_dict = dict()
            # učitavamo datoteku s embeddinzima tokena i spremamo ih u rječnik
            with open(path, 'r') as file:
                while(True):
                    line = file.readline()
                    if not line:
                        break
                    ind = line.find(" ")
                    token = line[0:ind]
                    #print(token)

                    #pdb.set_trace()

                    x = np.array(['1.1', '2.2', '3.3'])
                    y = x.astype(np.float)

                    vector_reps = np.array(line[ind+1::].split(" ")).astype(np.float)
                    embeddings_dict.update({token:vector_reps})
        
            # na osnovu embedding rječnika i tokena iz rječnika kreiramo
            # embedding matricu
            i = 2
            #pdb.set_trace()
            for token in self.stoi_dict.keys():
                if (token == "<UNK>" or token == "<PAD>"):
                    #print("Skipping:",token)
                    continue
                if (token in embeddings_dict):
                    #print("Ovdje:", token)
                    embedding_matrix[i] = embeddings_dict[token]
                i += 1
        
        # vraćamo ju u optimiziranom omotaču
        embedding_matrix_tensor = torch.FloatTensor(embedding_matrix)
        return torch.nn.Embedding.from_pretrained(embedding_matrix_tensor, freeze = freeze, padding_idx = 0)
        