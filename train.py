import argparse
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
from collections import OrderedDict
def save_checkpoint(vgg, arch, hidden_units, learning_rate, epochs, optimizer, save_dir, class_to_idx):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': vgg.state_dict(),
        'class_to_idx': class_to_idx
    }
    filename = f'{save_dir}/checkpoint.pth'
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--data_dir', type=str, help=' dataset directory')
parser.add_argument('--save_dir', type=str, default='/home/workspace/ImageClassifier', help='save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='Architecture (vgg13, vgg16, ...)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='hidden')
parser.add_argument('--epochs', type=int, default=10, help=' training epochs')
parser.add_argument('--gpu', action='store_true', help='GPU')
args = parser.parse_args()
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

train_transforms=transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
validation_transforms=transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
test_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)
valid_datasets= datasets.ImageFolder(valid_dir,transform=validation_transforms)
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_datasets,batch_size=32,shuffle=True)
validloader =torch.utils.data.DataLoader(valid_datasets,batch_size=32,shuffle=True)
vgg = getattr(models, args.arch)(pretrained=True)
for param in vgg.parameters():
    param.requires_grad = False
classifier = nn.Sequential(
     nn.Linear(25088, args.hidden_units),
     nn.ReLU(),
     nn.Dropout(0.5),
     nn.Linear(args.hidden_units, 102),
     nn.LogSoftmax(dim=1)
)
vgg.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(vgg.classifier.parameters(), lr=args.learning_rate)
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
vgg=vgg.to(device)
for epoch in range(args.epochs):
    running_loss = 0
    for inputs, labels in trainloader:  
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = vgg(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    validation_loss = 0.0
    accuracy = 0.0
    test_loss = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in validloader:  
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = vgg(inputs)
            validation_loss += criterion(outputs, labels).item()
            test_loss += criterion(outputs, labels).item()
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Epoch {epoch+1}/{args.epochs}.. ")
    print( f"Training loss: {running_loss/len(trainloader):.3f}")
    print(f"Validation loss: {validation_loss/len(validloader):.3f}")
    print( f"Validation Accuracy: {accuracy/len(validloader):.3f}")
    print(f"Testing Loss: {test_loss/len(testloader)}")
    print(f"Testing Accuracy: {test_accuracy/len(testloader)}")
save_checkpoint(vgg, args.arch, args.hidden_units, args.learning_rate, args.epochs, optimizer, args.save_dir, train_datasets.class_to_idx)