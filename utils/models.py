import torch.nn as nn

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
        x = self.classifier(x)
        return x
    
class OneHotClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(OneHotClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x