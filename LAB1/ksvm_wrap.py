from sklearn.svm import SVC
import numpy as np

class KSVMWrap:
    # u initu također provodimo fittanje SVM-a
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto', kernel="linear"):
        super().__init__()
        # inicijaliziraj scikitov svm
        self.svm_model = SVC(C= param_svm_c, gamma = param_svm_gamma, kernel=kernel)
        self.svm_model.fit(X, Y_)
    
    # predviđa i vraća indekse razreda podataka X
    def predict(self, X):
        return self.svm_model.predict(X)
    
    # vraća klasifikacijske mjere podatka X
    # potrebno prilikom računanja prosječne preciznosti
    def get_scores(self, X):
        self.svm_model.decision_function(X)
    
    # indeksi podataka koji su odabrani kao potporni vektori
    # njih "duplamo" u grafu
    def get_support(self):
        return self.svm_model.support_
    
    
    
    