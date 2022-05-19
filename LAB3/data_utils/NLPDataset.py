# razred služi za učitavanje podataka te njihovo spremanje u memoriju i dohvaćanje, te za operacije za koje nam je potreban cijeli skup podataka, poput izgradnje vokabulara
# u ovom razredu učitavaš podatke, računaš njihove frekvencije te gradiš vocabulary
from .vocabulary import *
import csv
from .data_utils import get_frequencies_tokens, get_frequencies_labels
from .instance import Instance
import torch


class NLPDataset(torch.utils.data.Dataset):
    # moramo podržati predavanje vokabulara u konstruktoru
    # jer su valid i test Vocabulariyi jednkai kao
    # vocabi od train skupa
    def __init__(self, path, data_voc: Vocabulary = None, label_voc: dict = None, vocab_max_size = 100000, vocab_min_freq = 1):
        self.data_voc = data_voc
        self.label_voc = label_voc
        self.instances = []
        
        #učitavanje podataka
        with open(path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                token_list = row[0].split(" ")
                label = row[1]
                instance = Instance(token_list, label)
                self.instances.append(instance)
                
        # kreiranje rječnika, ako nisu zadani - ovo je slučaj za train set
        # valid i test set će imati zadane vokabulare
        if (data_voc == None):
            freq_dict = get_frequencies_tokens(self.instances)
            self.data_voc = Vocabulary(freq_dict, vocab_max_size, vocab_min_freq)
            print("Kreiram svoj data vocabulary")
        if (label_voc == None):
            self.label_voc = {"negative":0, "positive":1}
    
    def __len__(self):
        return len(self.instances)
    
    # dohvaća zadanu instancu, te ju pretvara u tenzor brojeva koristeći svoj rječnik
    def __getitem__(self, index):
        
        instance = self.instances[index]
        token_list = instance.token_list
        label = instance.label.strip()
        
        #pdb.set_trace()
        
        index_list_tensor = torch.tensor(self.data_voc.encode(token_list))
        label_index_tensor = torch.tensor(self.label_voc[label])
        
        return index_list_tensor, label_index_tensor
        