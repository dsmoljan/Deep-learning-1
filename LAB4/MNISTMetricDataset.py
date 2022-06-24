from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision
import random
import pdb

# Implementirajte metode _sample_positive i _sample_negative tako da njihove povratne vrijednosti odgovaraju indeksima uzorkovanih slika u listi self.images. # # za potrebe ove vježbe dovoljno je implementirati jednostavno uzorkovanje koje će za pozitivni primjer uzorkovati slučajnu sliku koja pripada istom razredu # kao sidro, a za negativni primjer slučajnu sliku koja pripada bilo kojem razredu različitom od razreda sidra.


class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))
        
        if (remove_class is not None):
            self.images = self.images[self.targets != remove_class]
            self.targets = self.targets[self.targets != remove_class]
            self.classes = list(range(1,10))

        # defaultdict je podvrsta dictionarya u kojem možemo reći
        # koja će se vrijednost pridružiti novom ključu koji još nema vrijednost
        # npr. ovdje koristimo defaultdict(list), što znači da će se novom ključu u rječniku
        # po defaultu pridružiti vrijednost []
        # npr. {a:[]}
        # ovdje valjda samo mapiramo indekse primjera na njihove oznake razreda
        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]
        # dakle, dict target2indices bi trebao imati 10 elemenata
        # ključ elementa - oznaka klase, value elementa - lista u kojoj su indeksi svih primjera koji pripadaju toj klasi
        # dakle npr. {0:[12,14,18,...888,901,903]}

    # slučajno odabire jedan negativni primjer u odnosu na anchor na danom indeksu
    def _sample_negative(self, anchor_index):
        anchor_class = self.targets[anchor_index].item()
        # moramo odabrati bilo koji primjer iz bilo koje klase koja nije anchor_class
        negative_classes = self.classes.copy()
        negative_classes.remove(anchor_class)
        negative_class = random.choice(negative_classes)
        # dohvati indekse svih primjera koji pripadaju toj klasi - to radi self.target2indices[negative_class]
        # i iz svih tih indeksa nasumično odaberi jedan
        negative_sample_index = random.choice(self.target2indices[negative_class])
        return negative_sample_index

    def _sample_positive(self, anchor_index):
        #pdb.set_trace()
        anchor_class = self.targets[anchor_index].item()
        positive_class_examples = self.target2indices[anchor_class].copy()
        # iz popisa svih primjera koji pripadaju pozitivnoj klasi izbacujemo anchor primjer, jer
        # ne želimo njega odabrati kao pozitivni primjer
        positive_class_examples.remove(anchor_index)
        positive_sample_index = random.choice(positive_class_examples)
        return positive_sample_index

    # dohvaća trojku (anchor, positive, negative), gdje su (positive, negative) slučajno odabrani
    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()

        # ako se radi o evaluaciji, onda vrati samo anchor
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            #pdb.set_trace()
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)