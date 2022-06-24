import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import PairwiseDistance


# ovo nam predstavlja modificirani CNN sloj
class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        # stavi padding = 1 ako nešto neće dobro raditi
        # formula za računanje izlaza konvolucijskog sloja: https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer

        self.add_module("batch_norm", nn.BatchNorm2d(num_maps_in))
        self.add_module("relu", torch.nn.ReLU())
        self.add_module("conv", nn.Conv2d(in_channels=num_maps_in, out_channels=num_maps_out, kernel_size=k, stride=1,
                                          padding=0, bias=True))


class SimpleMetricEmbedding(nn.Module):
    # emb_size je samo broj kanala, ništa posebno
    # input channels
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # ulazni tenzor = [N, 1, 28, 28]
        self.BNR_1 = _BNReluConv(input_channels, emb_size)
        # ulaz u pool1: [N, 1, 26, 26]
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # ulaz u BNR_2 : [N, C, 12, 12]
        self.BNR_2 = _BNReluConv(emb_size, emb_size)
        # ulaz u pool2: [N, C, 10, 10]
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # ulaz u BNR_3: [N, C, 4, 4]
        self.BNR_3 = _BNReluConv(emb_size, emb_size)
        # ulaz u avg_pool: [N, C, 2, 2]
        # provjeri dobro jel ovdje oke koristiti avg_pool
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # izlaz iz avg pool: [N, C, 1, 1]

    # ovo je ekvivalent forward metodi u prijašnjim modelima
    # samo nam sad izlaz modela nije klasa, nego embedding (featurei) dane slike
    # pa se zato metoda zove get_features
    def get_features(self, img):
        # EMB_SIZE == broj kanala
        # BATCH_SIZE == N
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE --> dakle izlaz treba biti BATCH_SIZE x EMB_SIZE x 1 x 1, koji onda samo squeezas

        h1 = self.BNR_1(img)
        p1 = self.pool1(h1)
        h2 = self.BNR_2(p1)
        p2 = self.pool2(h2)
        h3 = self.BNR_3(p2)
        x = self.avg_pool(h3)

        # squeeze da iz [N, C, 1, 1] dobijemo [N, C]
        x = x.squeeze(3)
        x = x.squeeze(2)
        # alternativni način za micanje te zadnje dvije dimenzije: x = x.reshape((img.size(dim=0), self.emb_size))
        return x

    def loss(self, anchor, positive, negative, alpha=1.0, p=2):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        # za udaljenost između primjera koristi ovo https://pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html
        # defaultna vrijednost p-a = 2, pa ovo postaje obična L2 norma
        pdist = PairwiseDistance(p)

        #loss = relu((torch.norm((a_x - p_x), dim=1, p=p) - torch.norm((a_x - n_x), dim=1, p=p) + alpha), 0)
        loss = relu((pdist(a_x, p_x) - pdist(a_x, n_x) + alpha), 0)
        mean_loss = torch.mean(loss)
        # vraćamo mean loss jer je loss prije meana tenzor u obliku [64], tj. svaki indeks u tenzoru je
        # loss jednog primjera iz batcha
        return mean_loss
