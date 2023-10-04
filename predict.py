import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
from torch import nn
import json
import argparse

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet':
        model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Linear(25088, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(checkpoint['hidden_units'], 102),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['class_to_idx'], {v: k for k, v in checkpoint['class_to_idx'].items()}

def process_image(image):
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = image_transforms(image)
    return img

def predict(image_path, model, topk=5):
    model.eval()
    img = Image.open(image_path)
    img = process_image(img)
    img = img.unsqueeze(0)
    
    img = img.to(device)
    
    with torch.no_grad():
        output = model(img)
        ps = F.softmax(output, dim=1)
        top_p, top_class = ps.topk(topk, dim=1)
        
    return top_p, top_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict flower')
    parser.add_argument('--image_path', type=str, default='/home/workspace/ImageClassifier/flowers/test/100/image_07899.jpg', help='Path to image')
    parser.add_argument('--checkpoint_path', type=str, default='/home/workspace/ImageClassifier/checkpoint.pth', help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Top K')
    parser.add_argument('--gpu', action='store_true', help='GPU')
    parser.add_argument('--class_mapping', type=str, default='/home/workspace/ImageClassifier/cat_to_name.json', help='Path to JSON file')
    args = parser.parse_args()
    topk = args.top_k
    model, class_to_idx, idx_to_class = load_checkpoint(args.checkpoint_path)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    image_path = args.image_path
    
    top_p, top_class = predict(image_path, model, topk)
    top_class_list = top_class.squeeze().tolist()
    
    with open(args.class_mapping, 'r') as f:
        cat_to_name = json.load(f, strict=False)
    for i in range(topk):
        print("Predicted flower class:", i+1 , cat_to_name[idx_to_class[top_class_list[i]]])
    print("Probabilities:", top_p)
