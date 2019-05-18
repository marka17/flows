import torch
from torch import nn
from torch.nn.parameter import Parameter

class RealNVP(nn.Module):
    def __init__(self,
                 scale_network: nn.Module,
                 translation_network: nn.Module,
                 masks: torch.Tensor,
                 prior: torch.distributions.Distribution) -> None:
        super(RealNVP, self).__init__()

        self.num_layers = len(masks)
        self.masks = Parameter(masks, requires_grad=False)
        self.prior = prior

        self._scale_networks = nn.ModuleList([scale_network for _ in range(self.num_layers)])
        self._translation_networks = nn.ModuleList([translation_network for _ in range(self.num_layers)])

    def g(self, x: torch.Tensor) -> torch.Tensor:
        for mask, s, t in zip(self.masks, self._scale_networks, self._translation_networks):
            masked_x = x * mask

            scaled_x = s(masked_x)
            translated_x = t(masked_x)

            temp = x * torch.exp(scaled_x) + translated_x
            x = temp * (1 - mask) + masked_x

        return x

    def f(self, x: torch.Tensor) -> torch.Tensor:
        z, log_jacobian = x, torch.zeros(x.size(0))

        loader = list(zip(self.masks, self._scale_networks, self._translation_networks))
        for mask, s, t in reversed(loader):
            masked_z = z * mask

            scaled_z = s(masked_z)
            translated_z = t(masked_z)

            temp = torch.exp(-scaled_z) * (z - translated_z)
            z = temp * (1 - mask) + masked_z

            log_jacobian = log_jacobian - (scaled_z * (1 - mask)).sum(dim=-1)

        return z, log_jacobian

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_jacobian = self.f(x)

        return self.prior.log_prob(z) + log_jacobian

    def sample(self, batch_size: int) -> torch.Tensor:
        z = self.prior.sample(torch.Size((batch_size, )))

        return self.g(z)
