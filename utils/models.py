# HML Classifier code and get_constr_out 
# from Constrained Feed-Forward Neural Network for HML
# (Coherent Hierarchical Multi-Label Classification Networks - GPL-3.0 License)
# https://github.com/EGiunchiglia/C-HMCNN

import torch
import torch.nn as nn

def get_constr_out(x, R):
    """Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R"""
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
    return final_out

class FathomNetModel(nn.Module):
    def __init__(
            self,
            encoder,
            classifier,
    ):
        super(FathomNetModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        x = self.encoder(x)
        outs = self.classifier(x)
        return outs
    
class OneHotClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(OneHotClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
class MultiHeadClassifier(nn.Module):
    def __init__(
            self, 
            classifier_type, 
            num_classifiers, 
            features_dim, 
            output_dim
            ):
        super().__init__()

        self.classifiers = nn.ModuleList()  # Use ModuleList for proper parameter registration
        for _ in range(num_classifiers):
            if "one_hot" in classifier_type:
                classifier = OneHotClassifier(features_dim, output_dim)
            elif classifier_type == "hml":
                classifier = ConstrainedFFNNModel(
                    input_dim=features_dim,
                    hidden_dim=[features_dim],
                    output_dim=output_dim,
                    dropout=0.7,
                )
            else:
                raise ValueError(f"Unknown classifier type: {classifier_type}")
            
            self.classifiers.append(classifier)
    
    def forward(self, x):
        return torch.stack([classifier(x) for classifier in self.classifiers])
    
class ConstrainedFFNNModel(nn.Module):
    """C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, non_lin="relu"):
        super(ConstrainedFFNNModel, self).__init__()
        fc = []

        if len(hidden_dim) == 0:
            fc.append(nn.BatchNorm1d(input_dim, affine=False))
            fc.append(nn.Linear(input_dim, output_dim))
        else:
            fc.append(nn.BatchNorm1d(input_dim, affine=False))
            fc.append(nn.Linear(input_dim, hidden_dim[0]))
            for i in range(len(hidden_dim)):
                if i == len(hidden_dim) - 1:
                    fc.append(nn.BatchNorm1d(hidden_dim[i], affine=False))
                    fc.append(nn.Linear(hidden_dim[i], output_dim))
                else:
                    fc.append(nn.BatchNorm1d(hidden_dim[i], affine=False))
                    fc.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))

        self.fc = nn.ModuleList(fc)
        self.sigmoid = torch.nn.Sigmoid()

        self.drop = nn.Dropout(dropout)

        if non_lin == "tanh":
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()

    def forward(self, x):
        module_len = len(self.fc)
        for i in range(module_len):
            if i == module_len - 1:
                x = self.fc[i](x)
                x = self.sigmoid(x)
            else:
                if i % 2 == 0:
                    x = self.f(self.fc[i](x))
                else:
                    x = self.f(self.fc[i](x))
                    x = self.drop(x)
        return x