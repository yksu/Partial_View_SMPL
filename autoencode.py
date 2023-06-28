import torch
import torchvision
import torch.nn as nn

### Simple model for Auto Encoder
class Auto_Encoder(nn.Module):

    def __init__(self):

        super(Auto_Encoder, self).__init__()

        # Encoder
        self.Encoder = nn.Sequential(
            nn.Linear(36300, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        # Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 36300),
            nn.Sigmoid()
        )

    def forward(self, input):

        #code = input.view(input.size(0), -1)
        code = self.Encoder(input)
        output = self.Decoder(code)
        return output