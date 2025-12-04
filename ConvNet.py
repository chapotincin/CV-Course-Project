import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractorNet(nn.Module):
    def __init__(self):
        super(FeatureExtractorNet, self).__init__()
        
        # Define layers for a galaxy's feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ReLU(True)
        )

        # Define final convolutional output size
        self.flattened_size = 64 * 53 * 53
        
    # Extract the features from a galaxy
    def forward(self, X):
        X = self.features(X)
        X = torch.flatten(X, 1)

        return X

class ClassifierNet(nn.Module):

    def __init__(self, num_classes, fe=None):
        super().__init__()
        self.num_classes = num_classes
        self.fe = fe if fe else FeatureExtractorNet()

        # Define a galaxy classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fe.flattened_size, 90),
            nn.Dropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(90, 20),
            nn.Dropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(20, num_classes)
        )

    # Classify a galaxy as either elliptical or spiral
    def forward(self, x):
        features = self.fe(x)
        return self.classifier(features)
