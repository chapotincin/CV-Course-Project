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


class GalaxyTypeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = FeatureExtractorNet()

        # Define a galaxy classifier
        self.classifier = nn.Linear(self.fe.flattened_size, 2)

    # Classify a galaxy as either elliptical or spiral
    def forward(self, x):
        features = self.fe(x)

        return self.classifier(features)


class RoundnessNet(nn.Module):
    def __init__(self, num_roundness_classes):
        super().__init__()
        self.fe = FeatureExtractorNet()

        # Define a roundness measurer
        self.measurer = nn.Linear(self.fe.flattened_size, num_roundness_classes)

    # Measure the roundness of an elliptical galaxy
    def forward(self, x):
        features = self.fe(x)

        return self.measurer(features)


class NumSpiralArmsNet(nn.Module):
    def __init__(self, num_arm_classes):
        super().__init__()
        self.fe = FeatureExtractorNet()

        # Define an arm counter
        self.counter = nn.Linear(self.fe.flattened_size, num_arm_classes)

    # Count the number of arms of a spiral galaxy
    def forward(self, x):
        features = self.fe(x)

        return self.counter(features)