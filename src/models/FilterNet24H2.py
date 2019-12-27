import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def conv_layer(window, ks=3, dilation=1):
    return nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=ks, bias=False, dilation=dilation),
        nn.AdaptiveAvgPool1d(window),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))

class FilterNet24H2(nn.Module):
    def __init__(self, out_sz, window=24, filters=[1, 2, 3, 4, 5, 6]):
        super(FilterNet24H2).__init__()

        # TODO: Use the filters arg to generate the conv_layers dynamically
        # Wavenet model layers
        self.c1a = conv_layer(window=window // 2, ks=1, dilation=1)
        self.c1b = conv_layer(window=window // 4, ks=1, dilation=2)
        self.c2a = conv_layer(window=window // 2, ks=2, dilation=1)
        self.c2b = conv_layer(window=window // 4, ks=2, dilation=2)
        self.c3a = conv_layer(window=window // 2, ks=3, dilation=1)
        self.c3b = conv_layer(window=window // 4, ks=3, dilation=2)
        self.c4a = conv_layer(window=window // 2, ks=4, dilation=1)
        self.c4b = conv_layer(window=window // 4, ks=4, dilation=2)
        self.c5a = conv_layer(window=window // 2, ks=5, dilation=1)
        self.c5b = conv_layer(window=window // 4, ks=5, dilation=2)
        self.c6a = conv_layer(window=window // 2, ks=6, dilation=1)
        self.c6b = conv_layer(window=window // 4, ks=6, dilation=2)

        num_wave_outputs = (len(filters) * (window // 2)) + (len(filters) * (window // 4))

        # Final layer
        self.f = Flatten()
        self.lin = nn.Linear(sizes[-2] + num_wave_outputs, out_sz, bias=False)

        self.sizes = sizes
        self.num_wave_outputs = num_wave_outputs

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_window, x_cat, x_cont):
        # TODO: Use the filters arg to generate the conv_layers dynamically
        # Wavenet model
        self.f1a = self.c1a(x_window)
        self.f1b = self.c1b(self.f1a)
        self.f2a = self.c2a(x_window)
        self.f2b = self.c2b(self.f2a)
        self.f3a = self.c3a(x_window)
        self.f3b = self.c3b(self.f3a)
        self.f4a = self.c4a(x_window)
        self.f4b = self.c4b(self.f4a)
        self.f5a = self.c5a(x_window)
        self.f5b = self.c5b(self.f5a)
        self.f6a = self.c6a(x_window)
        self.f6b = self.c6b(self.f6a)
        self.ffc = torch.cat([self.f1a, self.f1b, self.f2a, self.f2b,
                              self.f3a, self.f3b, self.f4a, self.f4b,
                              self.f5a, self.f5b, self.f6a, self.f6b, ], 2)

        self.fc = torch.cat([self.ffc], 2)
        self.flin = self.lin(self.f(self.fc))
        return self.flin


class FilterNetDataset(torch.utils.data.Dataset):
    def __init__(self, x_window, x_cat, x_cont, y):
        self.x_window = x_window
        self.x_cat = x_cat
        self.x_cont = x_cont
        self.y = y

    def __getitem__(self, idx): return [self.x_window[idx], self.x_cat[idx], self.x_cont[idx]], self.y[idx]
    def __len__(self): return max(len(self.x_window), len(self.x_cat), len(self.x_cont))
