import time
import torch
import torch.nn as nn

class Counter(nn.Module):
    def __init__(self, model):
        super(Counter, self).__init__()
        self.model = model
        self.activation_counts = 0
        self.matmul_counts = 0
        self._register_hooks()

    def _register_hooks(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(self._hook)
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(self._hook)

    def _hook(self, module, input, output:torch.Tensor):
        if isinstance(module, nn.ReLU):
            self.activation_counts += output.flatten(start_dim=1).size(1)
        # Count matrix multiplications for Linear layers
        elif isinstance(module, nn.Linear):
            self.matmul_counts += input[0].size(0) * module.weight.size(0) * module.weight.size(1)
        # Count matrix multiplications for Conv2d layers
        elif isinstance(module, nn.Conv2d):
            output_elements = output.numel()  # Total elements in the output tensor
            kernel_elements = module.weight.size(2) * module.weight.size(3) * module.in_channels
            self.matmul_counts += output_elements * kernel_elements

    def forward(self, x):
        return self.model(x)

    def reset_counts(self):
        self.activation_counts = 0
        self.matmul_counts = 0

    def get_counts(self):
        return self.activation_counts, self.matmul_counts

class denseBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layers = []
        for i in range(4):
            layer = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(64+i*32,4*32,1,1,0,bias=False),
                nn.ReLU(),
                nn.Conv2d(4*32,32,3,1,1,bias=False)
            )
            layers.append(layer)
        self.layer = nn.Sequential(*layers)

    def forward(self, init_features):
        features = [init_features]
        for layer in self.layer:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class residualBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(64,96,3,1,1,bias=False)
        self.conv2 = nn.Conv2d(96,196,3,1,1,bias=False)
        self.relu = nn.ReLU()
        self.skip = nn.Conv2d(64,196,1,1,0,bias=False)

    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        x = x + identity
        x = self.relu(x)
        return x
            

def main() -> None:
    # desBlock = denseBlock().cuda()
    # counter = Counter(desBlock).cuda()
    # counter(torch.randn([1,64,32,32]).cuda())
    # print('desBlock param: %d' % sum(p.numel() for p in desBlock.parameters()))
    # print('desBlock acts: %d' % counter.get_counts()[0])
    # print('desBlock muls: %d' % counter.get_counts()[1])

    resBlock = residualBlock().cuda()
    counter = Counter(resBlock).cuda()
    counter(torch.randn([1,64,32,32]).cuda())
    print('resBlock param: %d' % sum(p.numel() for p in resBlock.parameters()))
    print('resBlock acts: %d' % counter.get_counts()[0])
    print('resBlock muls: %d' % counter.get_counts()[1])

    iteration = 1000
    x = torch.randn([128,64,32,32]).cuda()

    # desBlock.eval()
    # t_time = 0.0
    # for i in range(iteration):
    #     s_time = time.time()
    #     y = desBlock(x)
    #     t_time += time.time() - s_time
    # print('desBlock time: %.3fms' % (t_time / iteration * 1000))

    resBlock.eval()
    t_time = 0.0
    for i in range(iteration):
        s_time = time.time()
        y = resBlock(x)
        t_time += time.time() - s_time
    print('resBlock time: %.3fms' % (t_time / iteration * 1000))


if __name__=='__main__':
    main()