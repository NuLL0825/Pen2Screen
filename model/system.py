import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import re
import torch.nn.functional as F


# CRNN Model
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True) if leakyRelu else nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling2', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling3', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)
        return output
    
class LatexTokenizer:
    def __init__(self, symbol_to_index):
        self.symbol_to_index = symbol_to_index

    def tokenize(self, expression):
        return re.findall(r'\\[a-zA-Z]+[\(\)\|\_]?|[a-zA-Z0-9]+|\^|_|[+\-*/=(){}]', expression)

    def encode(self, expression):
        tokens = self.tokenize(expression)
        return [self.symbol_to_index.get(token, self.symbol_to_index['<UNK>']) for token in tokens]

class LatexPostProcessor:
    def __init__(self, index_to_symbol):
        self.index_to_symbol = index_to_symbol

    def decode(self, output_indices):
        return ''.join([self.index_to_symbol.get(str(index), '<UNK>') for index in output_indices])
    
# latex_index_path = "latex.json"

# # Load LaTeX token mapping
# with open(latex_index_path) as f:
#     ltx_index = json.load(f)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer = LatexTokenizer(ltx_index['symbol_to_index'])
# post_processor = LatexPostProcessor(ltx_index['index_to_symbol'])
    
# num_classes = len(ltx_index['symbol_to_index'])

# model = CRNN(imgH=32, nc=1, nclass=num_classes, nh=256).to(device)

# model.load_state_dict(torch.load("fine_tuned_CRNN.pth" , weights_only=True, map_location=torch.device('cpu')))

# model.eval()


# # Load the image_
# image_path = "segmented_chars/char_000_001.png"
# image = Image.open(image_path)

# # Visualize the original image
# plt.figure(figsize=(6, 4))
# plt.imshow(image, cmap='gray')  # Use cmap='gray' for grayscale images
# plt.title("Original Image")
# plt.axis('off')  # Hide axes
# plt.show()

# # Preprocess the image
# preprocess = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.Resize((32, 128)),  # Match model input size
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))  # Normalize to match training
# ])
# input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# # Predict
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# input_tensor = input_tensor.to(device)
# output = model(input_tensor)

# # Apply softmax to get probabilities
# probabilities = F.softmax(output, dim=2)  # Shape: [T, B, num_classes]

# # Get predicted class indices and their probabilities
# predicted_indices = torch.argmax(probabilities, dim=2)  # Shape: [T, B]
# predicted_confidences, _ = torch.max(probabilities, dim=2)  # Shape: [T, B]

# # Decode predictions and print confidence levels
# for t, (index, confidence) in enumerate(zip(predicted_indices[:, 0], predicted_confidences[:, 0])):
#     decoded_char = post_processor.index_to_symbol.get(str(index.item()), '<UNK>')
#     if decoded_char == '<PAD>':
#         break  # Stop processing further tokens
#     print(f"Character: {decoded_char}, Confidence: {confidence.item():.4f}")