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

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class TauEnsembleEfficientNet(nn.Module):
    def __init__(self, base_net, tau_arr, num_classes, device):
        super(TauEnsembleEfficientNet, self).__init__()
        self.base_net = base_net
        self.base_net.to(device)
        self.tau_arr = tau_arr
        self.num_classes = num_classes
        self.device = device

        self.weights = list(base_net._fc.parameters())[0].data.clone()
        self.normB = torch.norm(self.weights, 2, 1)
        self.ensemble_classifier = LocalLinear(num_classes, len(tau_arr),
                                               init_val=None).to(device)
        self.out_act = nn.ReLU()
                            
    def parameters(self):
        for param in self.ensemble_classifier.parameters():
            yield param

    def forward(self, x):
        with torch.no_grad():
            # self.base_net(x.to(self.device))
            rep = self.base_net._dropout(self.base_net._avg_pooling(
                self.base_net.extract_features(
                    x.to(self.device))).flatten(start_dim=1)).squeeze()

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

        x = self.out_act(self.ensemble_classifier(ensemble_logit).squeeze())
        return x

class TauBaggingEfficientNet:
    def __init__(self, base_net, tau_arr, device):
        self.base_net = base_net.to(device)
        self.tau_arr = tau_arr
        self.device = device

        self.weights = list(base_net._fc.parameters())[0].data.clone()
        self.normB = torch.norm(self.weights, 2, 1)

    def predict(self, x):
        rep = self.base_net._dropout(self.base_net._avg_pooling(
            self.base_net.extract_features(x.to(self.device))) \
            .flatten(start_dim=1)).squeeze()

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
            if ensemble_logit is None:
                ensemble_logit = logit
            else:
                ensemble_logit = torch.dstack((ensemble_logit, logit))
        return torch.mode(ensemble_logit.argmax(1), 1)[0]


class TauLogitEnsembleEfficientNet(nn.Module):
    def __init__(self, base_net, tau_arr, num_classes, device):
        super(TauLogitEnsembleEfficientNet, self).__init__()
        self.base_net = base_net
        self.base_net.to(device)
        self.tau_arr = tau_arr
        self.num_classes = num_classes
        self.device = device

        self.weights = list(base_net._fc.parameters())[0].data.clone()
        self.normB = torch.norm(self.weights, 2, 1)
        self.ensemble_classifier = nn.Linear(len(tau_arr)*num_classes, 
                                             num_classes).to(device)
        self.out_act = nn.Sigmoid()

    def parameters(self):
        for param in self.ensemble_classifier.parameters():
            yield param

    def forward(self, x):
        with torch.no_grad():
            # self.base_net(x.to(self.device))
            rep = self.base_net._dropout(self.base_net._avg_pooling(
                self.base_net.extract_features(
                    x.to(self.device))).flatten(start_dim=1)).squeeze()

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
                    ensemble_logit = torch.hstack((ensemble_logit, logit))

        x = self.out_act(self.ensemble_classifier(ensemble_logit).squeeze())
        return x


class TauDivideAndConquerClassifier(nn.Module):
    def __init__(self, divider, classifier_arr):
        super(TauDivideAndConquerClassifier, self).__init__()
        self.divider = divider
        self.classifier_arr = classifier_arr

    def forward(self, x):
        div = self.divider(x)
        assert div >= 0 and div < len(classifier_arr)-1, \
            "not sufficient number of classifier"
        x = classifier_arr[div][x]
        return x
