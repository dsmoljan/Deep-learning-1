import dataclasses
from torch.nn.utils.rnn import pad_sequence
import pdb
import torch

# zadatak collate funkcije je nadopuniti duljine
# instanci znakom punjenja do duljine najdulje instance u batchu
def pad_collate_fn(batch, pad_value=0):
    texts = []
    labels = []
    #pdb.set_trace()
    texts, labels = zip(*batch) # Assuming the instance is in tuple-like for - mora biti, ovo su sad tenzori
    
    lengths = torch.tensor([len(text) for text in texts])
    
    padded_texts = (pad_sequence(texts, batch_first=True, padding_value=pad_value))
                    
    return padded_texts, torch.tensor(labels), lengths

# prima listu objekata Instance, vraća rječnik sa brojem frekvencija za svaku riječ iz rječnika
def get_frequencies_tokens(instance_list):
    freq_dict = dict()
    
    for instance in instance_list:
        for token in instance.token_list:
            if (token not in freq_dict):
                freq_dict.update({token:1})
            else:
                freq_dict.update({token:freq_dict[token]+1})
    
    return freq_dict
    
def get_frequencies_labels(instance_list):
    freq_dict = dict()
    
    for instance in instance_list:
        if (instance.label not in freq_dict):
            freq_dict.update({instance.label:1})
        else:
            freq_dict.update({instance.label:freq_dict[instance.label]+1})
    
    
    return freq_dict
