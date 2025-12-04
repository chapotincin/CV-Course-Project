from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections.abc import Sequence
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
#IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#uncomment this to use Fabian conv code
from ConvNet import FeatureExtractorNet, ClassifierNet
#comment to not use udays code
#from Conv2Net import ConvNet
import argparse
import numpy as np
import torchvision
from torch.nn import CrossEntropyLoss


def train(model, device, train_loader, optimizer, criterion, epoch, batch_size, num_classes, debug_log = False):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    total = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)


        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        # ======================================================================
        # Count correct predictions overall 
        correct_this_time = pred.eq(target.view_as(pred)).sum().item()
        correct += correct_this_time
        total_this_time = len(pred)
        total += total_this_time
        if (batch_idx % 10 == 0 and debug_log):
            print("Guess is:")
            print(torch.flatten(pred))
            print("Actual is:")
            print(torch.flatten(target))
            print(str(correct_this_time) + " were correct out of " + str(total_this_time) + " (batch " + str(batch_idx) + ")")
    
    train_loss = float(np.mean(losses))
    train_acc = correct / total
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        float(np.mean(losses)), correct, total,
        100. * correct / total))
    return train_loss, train_acc
    


def test(model, device, test_loader, criterion, epoch, num_classes, debug_log = False):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    total = 0
    similarity_matrix = [[0 for i in range(num_classes)] for j in range(num_classes)]
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            

            # Predict for data by doing forward pass
            output = model(data)
            
            # ======================================================================
            # Compute loss based on same criterion as training
            loss = criterion(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # Reshape the target for convenience.
            target = target.view_as(pred)

            # Fill the similarity matrix.
            for pair in zip(pred, target):
                similarity_matrix[pair[0]][pair[1]]+=1

            # ======================================================================
            # Count correct predictions overall 
            correct_this_time = pred.eq(target).sum().item()
            correct += correct_this_time
            total_this_time = len(pred)
            total += total_this_time
            if (batch_idx % 10 == 0 and debug_log):
                print("Guess is:")
                print(torch.flatten(pred).tolist())
                print("Actual is:")
                print(torch.flatten(target).tolist())
                print(str(correct_this_time) + " were correct out of " + str(total_this_time) + " (batch " + str(batch_idx) + ")")
    
    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / total

    print('Test set:  Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, total, accuracy))
    
    return test_loss, accuracy, similarity_matrix

def output_similarity_matrix(num_epochs, similarity_matrix=None, mode=None, classes=[]):
    plt.figure(figsize=(8, 8)) # Adjust figure size as needed
    plt.imshow(similarity_matrix, cmap='viridis', origin='upper')
    # 3. Add a color bar to interpret the similarity values
    plt.colorbar(label='Similarity Score')
    plt.title("Similarity Matrix")
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)

    plt.savefig(f"similarity(mode={FLAGS.mode},num_epochs={num_epochs},test={mode}).png", dpi=300)

def output_graphs(epochs=[], accuracies=[], losses=[], mode=None, colors=('b', 'r')):
    '''
    Saves a graph of the accuracies and losses with respect to the epoch to the current working directory.
    epochs: A list of all the epochs.
    accuracies: A list of the accuracy for each epoch. 
    losses: A list of the loss for each epoch.
    mode: The mode to label the graph with, usually either 'Test' or 'Train'.
    colors: A two-tuple of the matplotlib colors used for the two lines. The first element is the color of the first line, and the second is the color of the second.
    '''
    
    if (not isinstance(epochs, Sequence)):
        raise TypeError("Epochs must be a sequence.")
    if (not isinstance(accuracies, Sequence)):
        raise TypeError("Accuracies must be a sequence.")
    if (not isinstance(losses, Sequence)):
        raise TypeError("Losses must be a sequence.")
    if (len(epochs) != len(accuracies)):
        raise ValueError("Expected same number of epochs and accuracies.")
    if (len(epochs) != len(losses)):
        raise ValueError("Expected same number of epochs and losses.")
    if (len(colors) != 2):
        raise ValueError("Expected two colors!")
    

    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(str(mode) + (" " if mode else "") + 'Accuracy', color=colors[0])
    ax1.plot(epochs, accuracies, color=colors[0])

    ax2 = ax1.twinx()
    
    ax2.set_ylabel(str(mode) + (" " if mode else "") + 'Loss', color=colors[1])
    ax2.plot(epochs, losses, color=colors[1])

    plt.savefig(f"output(mode={FLAGS.mode},num_epochs={epochs[-1]},test={mode}).png", dpi=300)

def create_weighted_sampler(dataset):
    '''
    Returns a weighted sampler for the given data set.
    Counts the number of elements in each class and over-samples some of the 
    classes to ensure a statistically balanced number of each.
    Returns the number of classes and the sampler.
    '''

    targets = []
    for _, label in dataset.samples:
        targets.append(label)
    
    class_counts = torch.bincount(torch.tensor(targets))
    total_samples = sum(class_counts)
    
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    weights = [class_weights[label] for label in targets]
    
    sampler = WeightedRandomSampler(weights, num_samples=len(targets), replacement=True)
    return len(class_counts), sampler

def run_main(FLAGS):
    
    debug_log = FLAGS.debug_log
    feature_extractor_path = FLAGS.feature_extractor_path
    model_path = FLAGS.model_path
    
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor()
    ])
    
    # Load datasets for training and testing
    # Path to load train and test from.
    train_path = FLAGS.data_path + r'/train'
    valid_path = FLAGS.data_path + r'/test'
    # Load datasets.
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    valid_dataset = torchvision.datasets.ImageFolder(root=valid_path, transform=transform)
    # Create weighted samplers.
    num_train_classes, train_sampler = create_weighted_sampler(train_dataset)
    num_valid_classes, valid_sampler = create_weighted_sampler(valid_dataset)
    # Create the data loaders.
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=FLAGS.batch_size, num_workers=4)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=FLAGS.batch_size, num_workers=4)
    
    print("Identified " + str(num_train_classes) + " training classes and " + str(num_valid_classes) + " testing classes.")

    # Ensure that there's the same number of training and testing classes.
    if (num_train_classes != num_valid_classes):
        raise ValueError("Expected same number of train and test classes! Got " + str(num_train_classes) + " and " + str(num_valid_classes) + ".")
    num_classes = num_valid_classes
    
    # Load the feature extractor net, if so desired.
    fe = None
    if (FLAGS.load_feature_extractor):
        fe = FeatureExtractorNet()
        fe.load_state_dict(torch.load(feature_extractor_path, weights_only=True))
        fe.eval()
        for p in fe.parameters():
            p.requires_grad=False;

    
    # Initialize the model and send to device 
    model = ClassifierNet(num_classes, fe=fe).to(device)
    
    
    if (FLAGS.load_model):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

    # Set the criterion as normal, with no weights.
    criterion = nn.CrossEntropyLoss()

    # set optimizer to SGD with learning rate specified in config.
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    

    # Initialize variables to hold all the collected data from training the 
    # model. Some of these may not be used with certain command-line settings,
    # but it's simpler to just collect and ignore the data than to not collect 
    # it.
    best_accuracy = 0.0
    best_similarity_matrix = None
    best_fe = None
    best_model = None
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []
    epochs = []
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print("Epoch " + str(epoch) + ":")
        
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch, FLAGS.batch_size, num_classes, debug_log)
        
        test_loss, test_accuracy, test_similarity_matrix = test(model, device, valid_loader, criterion, epoch, num_classes, debug_log)
        
        # If the accuracy is the best, record it.
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_similarity_matrix = test_similarity_matrix
            # Take a deep copy of the best state of the feature extractor.
            # A deep copy is needed because it's a reference; just normal 
            # copying isn't enough!
            best_fe = copy.deepcopy(model.fe.state_dict())
            # Take a deep copy of the best state of the model.
            best_model = copy.deepcopy(model.state_dict())

        epochs.append(int(epoch))
        test_losses.append(float(test_loss))
        test_accuracies.append(float(test_accuracy))
        train_accuracies.append(float(train_accuracy))
        train_losses.append(float(train_loss))

        print("End of epoch " + str(epoch) + ".\n")
    
    print("accuracy is {:2.2f}".format(best_accuracy))
    print("Training and evaluation finished")
    
    torch.device('cpu')
    
    if (FLAGS.save_graphs):
        print("Producing output graphs")
        output_graphs(epochs=epochs, accuracies=train_accuracies, losses=train_losses, mode="Train")
        output_graphs(epochs=epochs, accuracies=test_accuracies, losses=test_losses, mode="Test")
        output_similarity_matrix(epochs[-1], best_similarity_matrix, mode="Test", classes=valid_dataset.classes)
        print("Graphs saved")

    if (FLAGS.save_feature_extractor):
        torch.save(best_fe, feature_extractor_path)
    if (FLAGS.save_model):
        torch.save(best_model, model_path)
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-3.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument('--debug_log',
                        action='store_true', 
                        default=False,
                        help='Increase logging for debugging.')
    parser.add_argument('--save_graphs',
                        action='store_true', 
                        default=False,
                        help='Saves output graphs to a file in the current working directory.')
    parser.add_argument('--save_feature_extractor',
                        action='store_true', 
                        default=False,
                        help='Saves the feature extractor model to a file in the current working directory.')
    parser.add_argument('--save_model',
                        action='store_true', 
                        default=False,
                        help='Saves the full model to a file in the current working directory.')
    parser.add_argument('--load_feature_extractor',
                        action='store_true', 
                        default=False,
                        help='Loads the feature extractor model from a file in the current working directory.')
    parser.add_argument('--load_model',
                        action='store_true', 
                        default=False,
                        help='Loads the full model from a file in the current working directory.')
    parser.add_argument('--data_path',
                        type=str,
                        default='/content/drive/MyDrive/galaxies/type',
                        help='Directory containing the /train and /test data folders.')
    parser.add_argument('--feature_extractor_path',
                        type=str,
                        default="feature_extractor.pth",
                        help='The path to the feature extractor\'s weights, for saving or loading.')
    parser.add_argument('--model_path',
                        type=str,
                        default="model.pth",
                        help='The path to the model\'s weights, for saving or loading.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    
