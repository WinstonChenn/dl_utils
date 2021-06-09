import torch, copy
import torch.nn as nn


class LocalLinear(nn.Module):
    def __init__(self, num_features, kernel_size, init_val=1.0, bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.num_features = num_features

        if init_val is None:
            self.weight = nn.Parameter(torch.randn(num_features, kernel_size, 1))
            self.bias = nn.Parameter(torch.randn(num_features, 1)) if bias else None
        else:
            self.weight = nn.Parameter(torch.ones(num_features, kernel_size, 1)*init_val)
            self.bias = nn.Parameter(torch.ones(num_features, 1)*init_val) if bias else None

    def forward(self, x):
        assert x.shape[1] == self.num_features, \
            f"input dimension 1 ({x.shape[1]}) != num_features({self.num_features})"
        assert x.shape[2] == self.kernel_size, \
            f"input dimension 2 ({x.shape[2]}) != kernel_size({self.kernel_size})"
        x = torch.matmul(x.unsqueeze(2),self.weight).squeeze(2)
        if self.bias is not None:
            x = x+self.bias

        return x


class TauEnsembleEfficientNet(nn.Module):
    def __init__(self, base_net, tau_arr, num_classes, device):
        super(TauEnsembleEfficientNet, self).__init__()
        self.base_net = base_net
        self.base_net.to(device)
        self.tau_arr = tau_arr
        self.num_classes = num_classes
        self.device = device
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook
        self.weights = list(base_net._fc.parameters())[0].data.clone()
        self.normB = torch.norm(self.weights, 2, 1)
        self.base_net._conv_head.register_forward_hook(
            get_activation('_conv_head'))
        self.ensemble_classifier = LocalLinear(num_classes, len(tau_arr),
                                               init_val=None).to(device)

    def parameters(self, only_trainable=True):
        for param in self.ensemble_classifier.parameters():
            yield param

    def forward(self, x):
        with torch.no_grad():
            self.base_net(x.to(self.device))
            rep = self.base_net._dropout(self.base_net._avg_pooling(
                self.base_net._bn1(self.activation['_conv_head']
                                .to(self.device)))).squeeze()

            ensemble_logit = None
            for tau in self.tau_arr:
                ws = self.weights.clone()
                for i in range(self.weights.size(0)):
                    ws[i] = ws[i] / torch.pow(self.normB[i], tau)
                fc = copy.deepcopy(self.base_net._fc)
                list(fc.parameters())[0].data = ws.to(self.device)
                list(fc.parameters())[1].data = torch.zeros(50).to(self.device)

                def classifier(rep):
                    return self.base_net._swish(fc(rep))
                logit = classifier(rep)
                assert logit.shape[1] == self.num_classes
                if ensemble_logit is None:
                    ensemble_logit = logit
                else:
                    ensemble_logit = torch.dstack((ensemble_logit, logit))
        x = self.ensemble_classifier(ensemble_logit).squeeze()
        return x
