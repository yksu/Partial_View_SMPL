import torch
import torchvision
import torch.nn as nn

### Simple model for Auto Encoder
class Auto_Encoder(nn.Module):

    def __init__(self):

        super(Auto_Encoder, self).__init__()

        # Encoder
        self.Encoder = nn.Sequential(
            nn.Linear(45900, 8192),
            nn.RReLU(),
            nn.Linear(8192, 4096),
            nn.RReLU(),
            nn.Linear(4096, 1024),
            nn.Sigmoid()
        )

        # Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.RReLU(),
            nn.Linear(4096, 8192),
            nn.RReLU(),
            nn.Linear(8192, 45900),
            nn.Tanh()
        )

    def forward(self, input):

        #code = input.view(input.size(0), -1)
        code = self.Encoder(input)
        output = self.Decoder(code)
        return output
