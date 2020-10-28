#%%
import torch 
import tensorly as tl
tl.set_backend('pytorch')
import tensorly.random, tensorly.decomposition
from abc import abstractmethod, ABC
import torch.distributions as td
Parameter = torch.nn.Parameter

class LowRankTensor(ABC):
    def __init__(self,
                 dims,
                 prior_type=None,
                 init_method='random',
                 **kwargs):

        self.eps = 1e-12

        self.dims = dims
        self.order = len(self.dims)
        self.prior_type = prior_type

        for x in kwargs:
            setattr(self, x, kwargs.get(x))

        self.trainable_variables = []

        self._build_factors()
        self._build_factor_distributions()
        self._build_low_rank_prior()

    def get_relative_mse(self, sample_tensor):
        return torch.norm(self.get_full() -
                              sample_tensor) / torch.norm(sample_tensor)

    def add_variable(self, initial_value):

        #add weight using torch interface
        new_variable = Parameter(torch.tensor(initial_value))

        self.trainable_variables.append(new_variable)

        return new_variable

    @abstractmethod
    def _build_factors(self):
        pass

    @abstractmethod
    def _build_factor_distributions(self):
        pass

    @abstractmethod
    def _build_low_rank_prior(self):
        pass

    @abstractmethod
    def get_full(self):
        pass

    @abstractmethod
    def sample_full(self):
        pass

    @abstractmethod
    def get_rank(self, threshold=1e-4):
        pass

    @abstractmethod
    def prune_ranks(self, threshold=1e-4):
        pass

    @abstractmethod
    def get_kl_divergence_to_prior(self):
        pass

    @abstractmethod
    def get_parameter_savings(self):
        pass


class CP(LowRankTensor):
    def __init__(self, dims, max_rank, **kwargs):

        self.max_rank = max_rank
        self.max_ranks = max_rank
        super().__init__(dims, **kwargs)
        self.tensor_type = 'CP'
        

    #@tf.function
    def get_full(self):
        return tl.kruskal_to_tensor((self.weights, self.factors))

    def get_rank_variance(self):
        return torch.square(torch.relu(self.rank_parameter))


    def estimate_rank(self, threshold=1e-4):

        return len(torch.where(self.get_rank_variance() > threshold))

    def prune_ranks(self, threshold=1e-5):

        mask = (self.get_rank_variance()>threshold).float()

        self.weights = torch.squeeze(mask)
        
    def get_parameter_savings(self):
        
        rank_difference = self.max_rank-self.estimate_rank()
        savings_per_dim = [rank_difference*x for x in self.dims]
        low_rank_savings = sum(savings_per_dim)

        tensorized_savings = np.prod(self.dims)-sum([self.max_rank*x for x in self.dims])

        return low_rank_savings,tensorized_savings

    def _random_init(self):

        random_init = tl.random.random_kruskal(self.dims,
                                               self.max_rank,
                                               random_state=getattr(
                                                   self, 'seed', None))

        if hasattr(self, 'target_norm'):
            curr_norm = torch.norm(
                tl.kruskal_to_tensor(
                    (torch.ones([self.max_rank]), random_init[1])))
            mult_factor = torch.pow(
                getattr(self, 'target_norm') / curr_norm, 1 / len(self.dims))
            scaled_factors = [mult_factor * x for x in random_init[1]]
            random_init = (torch.ones([self.max_rank]), scaled_factors)

        return random_init

    def _nn_init(self):

        raise NotImplementedError

        if hasattr(self,"target.stddev"):
            pass
        else:
            self.target.stddev = 0.05
    
        factor.stddev = torch.pow(
            self.target.stddev / torch.sqrt(1.0 * self.max_rank),
            1.0 / len(self.dims))
        self.factor.stddev = factor.stddev
        initializer_dist = td.TruncatedNormal(loc=0.0,
                                               scale=factor.stddev,
                                               low=-3.0 * factor.stddev,
                                               high=3.0 * factor.stddev)
        init_factors = (torch.ones([self.max_rank]), [
            initializer_dist.sample([x, self.max_rank]) for x in self.dims
        ])

        return init_factors

    def _build_factors(self):

        if hasattr(self, 'initialization_method'):
            if self.initialization_method == 'random':
                self.weights, self.factors = self._random_init()
            elif self.initialization_method == 'parafac':
                self.initialization_tensor = torch.reshape(self.initialization_tensor,self.dims)
                self.weights, self.factors = tl.decomposition.parafac(
                    self.initialization_tensor, self.max_rank, init='random')
            elif self.initialization_method == 'parafac_svd':
                self.initialization_tensor = torch.reshape(self.initialization_tensor,self.dims)
                self.weights, self.factors = tl.decomposition.parafac(
                    self.initialization_tensor, self.max_rank, init='svd')
            elif self.initialization_method == 'nn':
                self.weights, self.factors = self._nn_init()
            else:
                raise (ValueError("Initialization method not supported."))
        else:
            self.weights, self.factors = self._random_init()

        #convert all to tensorflow variable
        self.factors = [self.add_variable(x) for x in self.factors]



    def _build_factor_distributions(self):

        factor_scale_prior_multiplier = 1e-3

        factor_scales = [
            self.add_variable(factor_scale_prior_multiplier *
                              torch.ones(factor.shape)) for factor in self.factors
        ]

        self.factor_distributions = []

        for factor, factor_scale in zip(self.factors, factor_scales):
            self.factor_distributions.append(
                td.Independent(td.Normal(
                    loc=factor,
                    scale=factor_scale),
                                reinterpreted_batch_ndims=2))


    def _build_low_rank_prior(self):

        self.rank_parameter = Parameter(torch.sqrt(torch.tensor(self.get_rank_parameters_update())))

        self.factor_prior_distributions = []

        for x in self.dims:
            self.factor_prior_distributions.append(td.Independent(td.Normal(loc=torch.zeros([x, self.max_rank]),scale=self.rank_parameter),reinterpreted_batch_ndims=2))

    def sample_full(self):
        return tl.kruskal_to_tensor(
            (self.weights, [x.rsample() for x in self.factor_distributions]))

    def get_rank_parameters_update(self):
        def half_cauchy():

            M = torch.sum(torch.tensor([torch.sum(torch.square(x.mean) + torch.square(x.stddev),dim=0) for x in self.factor_distributions]),dim=0)

            D = 1.0 * sum(self.dims)

            update = (M - D * self.eta**2 + torch.sqrt(torch.square(M) + (2.0 * D + 8.0) * torch.square(self.eta) * M +torch.pow(self.eta, 4.0) * torch.square(D))) / (2.0 * D + 4.0)

            return update

        def log_uniform():

            M = torch.sum(torch.stack([torch.sum(torch.square(x.mean) + torch.square(x.stddev),
                              dim=0) for x in self.factor_distributions]),dim=0)

            D = 1.0 * (sum(self.dims) + 1.0)

            update = M / D

            return update

        if self.prior_type == 'log_uniform':
            return log_uniform()
        elif self.prior_type == 'half_cauchy':
            return half_cauchy()
        else:
            raise ValueError("Prior type not supported")

    def update_rank_parameters(self):

        rank_update = self.get_rank_parameters_update().detach()


        self.rank_parameter.data = torch.sqrt((1 - self.em_stepsize) * self.rank_parameter**2 +
                                   self.em_stepsize * rank_update)


    def get_rank(self, threshold=1e-4):
        return len(torch.where(self.get_rank_variance() > threshold))


    def get_kl_divergence_to_prior(self):

        kl_divergences = [td.kl_divergence(factor_dist, factor_prior_dist) for (factor_dist, factor_prior_dist) in zip(self.factor_distributions, self.factor_prior_distributions)]

        return torch.sum(torch.stack(kl_divergences))




dims = [50,50,50]

tensor = CP(dims=dims,max_rank=10,prior_type='log_uniform',em_stepsize=0.5)

tensor.get_rank_parameters_update()

tensor.update_rank_parameters()


tensor.sample_full().shape

full = tl.kruskal_to_tensor(tl.random.random_kruskal(shape=dims,rank=3))


log_likelihood_dist = td.Normal(0.0,0.1)


def log_likelihood():
    return -torch.sum(log_likelihood_dist.log_prob(full-tensor.get_full()))/100


def mse():
    return torch.norm(full-tensor.get_full())/torch.norm(full)

def loss():
    return log_likelihood(tensor.get_full())+tensor.get_kl_divergence_to_prior()

#%%

loss = log_likelihood

optimizer = torch.optim.SGD(tensor.trainable_variables,lr=0.01)


for i in range(1000):

    optimizer.zero_grad()

    loss_value = loss()

    loss_value.backward()

    optimizer.step()

#    tensor.update_rank_parameters()

    if i%100==0:
#        print('Loss ',loss())
        print('RMSE ',mse())
        print(tensor.estimate_rank())

#%%

sums = [torch.sum(torch.square(x.mean) + torch.square(x.stddev),
                    dim=0) for x in tensor.factor_distributions])


M = torch.sum(torch.tensor(,axis=0)
#%%
D = 1.0 * (sum(self.dims) + 1.0)

update = M / D

print(M.shape)
print(update.shape)
return torch.expand_dims(update, 0)


#%%
import torch

td = torch.distributions
import numpy as np
Parameter = torch.nn.Parameter

mean = Parameter(torch.zeros([100,5]))
std = Parameter(torch.ones([100,5]))
tmp = td.Normal.mean,std)
tmp.sample().shape

#%%
import torch
import numpy as np
td = torch.distributions

Parameter = torch.nn.Parameter

mean = Parameter(torch.zeros([1,1]))
mean = torch.zeros([1,1])

std = Parameter(0.5*torch.ones([1,1]))

tmp = td.Normal.mean,torch.ones(std.shape))

tmp.sample().shape

transforms = [td.transforms.AffineTransform(loc=0,scale=torch.sqrt(std))]

new_normal = td.TransformedDistribution(tmp,transforms)

np.std(new_normal.sample([10000]).numpy())

#td.Normal(loc=torch.zeros([x, self.max_rank]),scale=torch.sqrt(self.rank_parameter))
# %%


class Scale_Normal(td.Transform):

    def __init__(self,scale_squared,cache_size=0):

        self._cache_size = cache_size
        
        self.scale_squared = scale_squared
        self._inv = None
        if cache_size == 0:
            pass  # default behavior
        elif cache_size == 1:
            self._cached_x_y = None, None
        else:
            raise ValueError('cache_size must be 0 or 1')
        super(Scale_Normal, self).__init__()

    def _call(self,x):

        return x*torch.sqrt(self.scale_squared)

transforms = [Scale_Normal(std)]

new_normal = td.TransformedDistribution(tmp,transforms) 

np.std(new_normal.sample([10000]).numpy())
# %%
def loss():
    return torch.norm(new_normal.rsample([1000]))

weights = [std]

optimizer = torch.optim.SGD(weights,lr=0.0001)

#%%

for i in range(1000):
    optimizer.zero_grad()

    loss_value = loss()

    loss_value.backward(retain_graph=True)

    optimizer.step()


    if i%100==0:
        print(loss())

#%%
