import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class SplineLinearLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_knots=5, spline_order=3,
                 noise_scale=0.1, base_scale=1.0, spline_scale=1.0,
                 activation=torch.nn.SiLU, grid_epsilon=0.02, grid_range=[-1, 1],
                 standalone_spline_scaling=True):
        super(SplineLinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_knots = num_knots
        self.spline_order = spline_order
        self.grid_epsilon = grid_epsilon
        self.grid_range = grid_range
        self.standalone_spline_scaling = standalone_spline_scaling

        self.knots = self._calculate_knots(grid_range, num_knots, spline_order)
        self.base_weights = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.spline_weights = torch.nn.Parameter(torch.Tensor(output_dim, input_dim, num_knots + spline_order))
        if standalone_spline_scaling:
            self.spline_scales = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))

        self.noise_scale = noise_scale
        self.base_scale = base_scale
        self.spline_scale = spline_scale
        self.activation = activation()

        self._initialize_parameters()

    def _initialize_parameters(self):
        torch.nn.init.xavier_uniform_(self.base_weights, gain=torch.sqrt(torch.tensor(2.0)))
        noise = torch.rand(self.num_knots + 1, self.input_dim, self.output_dim) - 0.5
        self.spline_weights.data.copy_(self.spline_scale * self._initialize_spline_weights(noise))
        if self.standalone_spline_scaling:
            torch.nn.init.xavier_uniform_(self.spline_scales, gain=torch.sqrt(torch.tensor(2.0)))

    def _calculate_knots(self, grid_range, num_knots, spline_order):
        h = (grid_range[1] - grid_range[0]) / num_knots
        knots = torch.arange(-spline_order, num_knots + spline_order + 1) * h + grid_range[0]
        return knots.expand(self.input_dim, -1).contiguous()

    def _initialize_spline_weights(self, noise):
        return self._fit_curve_to_coefficients(self.knots.T[self.spline_order : -self.spline_order], noise)

    def _compute_b_splines(self, x):
        x = x.unsqueeze(-1)
        bases = ((x >= self.knots[:, :-1]) & (x < self.knots[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - self.knots[:, : -(k + 1)]) / (self.knots[:, k:-1] - self.knots[:, : -(k + 1)]) * bases[:, :, :-1] +
                     (self.knots[:, k + 1 :] - x) / (self.knots[:, k + 1 :] - self.knots[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def _fit_curve_to_coefficients(self, x, y):
        A = self._compute_b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def _scaled_spline_weights(self):
        return self.spline_weights * (self.spline_scales.unsqueeze(-1) if self.standalone_spline_scaling else 1.0)

    def forward(self, x):
        base_output = F.linear(self.activation(x), self.base_weights)
        spline_output = F.linear(self._compute_b_splines(x).view(x.size(0), -1),
                                 self._scaled_spline_weights.view(self.output_dim, -1))
        return base_output + spline_output

    @torch.no_grad()
    def _update_knots(self, x, margin=0.01):
        batch = x.size(0)
        splines = self._compute_b_splines(x).permute(1, 0, 2)
        orig_coeff = self._scaled_spline_weights.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        adaptive_knots = x_sorted[torch.linspace(0, batch - 1, self.num_knots + 1, dtype=torch.int64, device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.num_knots
        uniform_knots = torch.arange(self.num_knots + 1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

        knots = self.grid_epsilon * uniform_knots + (1 - self.grid_epsilon) * adaptive_knots
        knots = torch.cat([
            knots[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
            knots,
            knots[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
        ], dim=0)

        self.knots.copy_(knots.T)
        self.spline_weights.data.copy_(self._fit_curve_to_coefficients(x, unreduced_spline_output))




class DeepKAN(torch.nn.Module):
    """"""
    def __init__(self, input_dim, hidden_layers, num_knots=5, spline_order=3,
                 noise_scale=0.1, base_scale=1.0, spline_scale=1.0,
                 activation=torch.nn.SiLU, grid_epsilon=0.02, grid_range=[-1, 1]):
        super(DeepKAN, self).__init__()
        layers = [input_dim] + hidden_layers
        self.layers = torch.nn.ModuleList()
        for in_dim, out_dim in zip(layers, layers[1:]):
            self.layers.append(SplineLinearLayer(in_dim, out_dim, num_knots, spline_order,
                                                 noise_scale, base_scale, spline_scale,
                                                 activation, grid_epsilon, grid_range))

    def forward(self, x, update_knots=False):
        """"""
        for layer in self.layers:
            if update_knots:
                layer._update_knots(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """"""
        return sum(layer._regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)




class ChebyshevKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyshevKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.xavier_uniform_(self.cheby_coeffs, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  
        x = torch.tanh(x)
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
        y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)  
        y = y.view(-1, self.outdim)
        return y



class RBFLinear(nn.Module):
   def __init__(self, in_features, out_features, grid_min=-2., grid_max=2., num_grids=8, spline_weight_init_scale=0.1):
       super().__init__()
       self.grid_min = grid_min
       self.grid_max = grid_max
       self.num_grids = num_grids
       self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=False)
       self.spline_weight = nn.Parameter(torch.randn(in_features*num_grids, out_features)*spline_weight_init_scale)

   def forward(self, x):
       x = x.unsqueeze(-1)
       basis = torch.exp(-((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1))) ** 2)
       return basis.view(basis.size(0), -1).matmul(self.spline_weight)

class RBFKANLayer(nn.Module):
   def __init__(self, input_dim, output_dim, grid_min=-2., grid_max=2., num_grids=8, use_base_update=True, base_activation=nn.SiLU(), spline_weight_init_scale=0.1):
       super().__init__()
       self.input_dim = input_dim
       self.output_dim = output_dim
       self.use_base_update = use_base_update
       self.base_activation = base_activation
       self.spline_weight_init_scale = spline_weight_init_scale
       self.rbf_linear = RBFLinear(input_dim, output_dim, grid_min, grid_max, num_grids, spline_weight_init_scale)
       self.base_linear = nn.Linear(input_dim, output_dim) if use_base_update else None

   def forward(self, x):
       ret = self.rbf_linear(x)
       if self.use_base_update:
           base = self.base_linear(self.base_activation(x))
           ret = ret + base
       return ret

class RBFKAN(nn.Module):
   def __init__(self, layers_hidden, grid_min=-2., grid_max=2., num_grids=8, use_base_update=True, base_activation=nn.SiLU(), spline_weight_init_scale=0.1):
       super().__init__()
       self.layers = nn.ModuleList([RBFKANLayer(in_dim, out_dim, grid_min, grid_max, num_grids, use_base_update, base_activation, spline_weight_init_scale) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])])

   def forward(self, x):
       for layer in self.layers:
           x = layer(x)
       return x