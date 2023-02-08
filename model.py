# PROGRAMMER: EZZA ALI

from torchvision import models
from torch import nn, optim
from collections import OrderedDict
import torch
from utility_funcs import get_dataloaders, get_train_data, process_image
from workspace_utils import active_session
import matplotlib.pyplot as plt

def create_model(arch, hidden_units, classes_out):
    """
    Create required model having given pretrained arch and hidden layers in the classifier.
    INPUTS:
        string: arch
        list: hidden_units
    Returns a model.
    """
    print('\nCreating model...')
    model = models.__dict__[arch](pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    
    densenet_features_in = {
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet161': 2208,
        'densenet201': 1920
    }

    features_in = 0

    # Set features in from the pretrained NN
    if arch.startswith("vgg"):
        features_in = model.classifier[0].in_features
    elif arch.startswith("densenet"):
        features_in = densenet_features_in[arch]
    
    hidden_units.insert(0, features_in)
    
    # Set output equal to the number of classes
    
    print(f"Images are labeled with {classes_out} categories.")
    
    
    # Create hidden layers with given units
    od = OrderedDict()
    for i in range(0, len(hidden_units)-1):
        od['fc' + str(i + 1)] = nn.Linear(hidden_units[i], hidden_units[i + 1])
        od['relu' + str(i + 1)] = nn.ReLU()
        od['dropout' + str(i + 1)] = nn.Dropout(p=0.15)
    
    od['output'] = nn.Linear(hidden_units[i + 1], classes_out)
    od['softmax'] = nn.LogSoftmax(dim=1)

    classifier = nn.Sequential(od)
    print('Classifier: ',classifier)
    model.classifier = classifier
    
    return model

def load_my_checkpoint(checkpoint_path, gpu):
    
    print('\nLoading checkpoint...')
    
    if gpu:
        if torch.cuda.is_available():
            # Use GPU if it's available
            device = torch.device("cuda:0")
            checkpoint = torch.load(checkpoint_path) # loading for GPU
            print(f'Using {device} to load the checkpoint.')
        else:
            device = torch.device("cpu")
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage) # loading for CPU
            print(f'GPU is not available. Using {device} to load the checkpoint.')        
    else:
        device = torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage) # loading for CPU
        print(f'Using {device} to load the checkpoint.')
        
    
    classes_out = len(checkpoint['class_to_idx'])
    hu = checkpoint['hidden_units'][1:]
    
    # Create the model
    model = create_model(checkpoint['arch'], hu, classes_out)
    
    # Load model's state
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    criterion = nn.NLLLoss()
    lr = checkpoint['learning_rate']
    opt = checkpoint['optimizer_state']
    print(f"Arch:{checkpoint['arch']}, Hidden_Units:{hu}, Learning Rate:{lr}")
    
    optimizer1 = optim.Adam(model.classifier.parameters(), lr=lr)
    optimizer1.state_dict = optimizer1.load_state_dict(opt)
    model.to(device);
    
    return model, checkpoint

def predict(image_path, checkpoint_path, gpu, cat_to_name, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        Returns: top_k probabilities, top_k Classes
    '''
    
    model, checkpoint = load_my_checkpoint(checkpoint_path, gpu)
    
    if gpu:
        if torch.cuda.is_available():
            # Use GPU if it's available
            device = torch.device("cuda:0")
            checkpoint = torch.load(checkpoint_path) # loading for GPU
            print(f'Using {device} to load the checkpoint.')
        else:
            device = torch.device("cpu")
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage) # loading for CPU
            print(f'GPU is not available. Using {device} to load the checkpoint.')        
    else:
        device = torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage) # loading for CPU
        print(f'Using {device} to load the checkpoint.')
        
    # the code to predict the class from an image file
    model.eval();
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image.to(device)
    model.to(device)
    
    with torch.no_grad():
        output = model.forward(image.to(device))
        top_prob, top_labels = torch.topk(output, top_k)
        
        # Calculate the exponentials
        top_prob = top_prob.exp()
    
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    top_labels.cpu()
    for label in top_labels.cpu().numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_prob.cpu().numpy()[0], mapped_classes
    
def evaluate_model(model, loader, criterion, device):
    loss = 0
    accuracy = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss/len(loader), accuracy/len(loader)

def train_save_model(arch, data_dir, lr, epochs, gpu, save_dir, hidden_units, classes_out):
    
    # Create Model
    model = create_model(arch, hidden_units, classes_out) 
    
    print('\nTraining model...')
    if gpu:
        if torch.cuda.is_available():
            # Use GPU if it's available
            device = torch.device("cuda:0")
            print(f'Using {device} to train the model.')
        else:
            device = torch.device("cpu")
            print(f'GPU is not available. Using {device} to train the model.')        
    else:
        device = torch.device("cpu")
        print(f'Using {device} to train the model.')
        
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.to(device);
    
    # train
    trainloader, validloader, testloader = get_dataloaders(data_dir)
    
    with active_session():
        steps = 0
        running_loss = 0
        print_every = 10

        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                #pint(device)
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {test_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()
    
    # Test the model 
    print('\nTesting model...')
    model.eval()
    test_loss, test_accuracy = evaluate_model(model, testloader, criterion, device)
    print ("\nTest Loss: {:.3f}.. ".format(test_loss),
           "Test Accuracy: {:.3f}".format(test_accuracy))
    
    # Save the checkpoint 
    print('\nSaving model...')
    train_data = get_train_data(data_dir)
    model.class_to_idx = train_data.class_to_idx
    torch.save({'arch': arch,
                'hidden_units': hidden_units,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx,
            'optimizer_state': optimizer.state_dict(),
            'learning_rate': lr,
            'epochs': epochs}, 
            save_dir+'/'+arch+'_checkpoint.pth')
    