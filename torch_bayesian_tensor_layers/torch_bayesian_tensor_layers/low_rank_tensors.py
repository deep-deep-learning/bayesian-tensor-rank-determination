#%%
import torch 
import tensorly as tl
tl.set_backend('pytorch')
import tensorly.random, tensorly.decomposition
from abc import abstractmethod, ABC
import torch.distributions as td
Parameter = torch.nn.Parameter
import numpy as np
from truncated_normal import TruncatedNormal

class LowRankTensor(torch.nn.Module):
    def __init__(self,
                 dims,
                 prior_type=None,
                 init_method='random',
                 **kwargs):

        super(LowRankTensor, self).__init__()

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

        self.trainable_variables = torch.nn.ParameterList(self.trainable_variables)

    def get_relative_mse(self, sample_tensor):
        return torch.norm(self.get_full() -
                              sample_tensor) / torch.norm(sample_tensor)

    def add_variable(self, initial_value,trainable=True):

        #add weight using torch interface
        new_variable = Parameter(initial_value.clone().detach(),requires_grad=trainable)

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
    def get_rank(self, threshold=1e-5):
        pass

    @abstractmethod
    def prune_ranks(self, threshold=1e-5):
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


    def estimate_rank(self, threshold=1e-5):

        return int(sum(sum(self.get_rank_variance() > threshold)))

    def prune_ranks(self, threshold=1e-5):

        mask = (self.get_rank_variance()>threshold).float()

        mask.to(self.rank_parameter.device)

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


        if hasattr(self,"target_stddev"):
            pass
        else:
            self.target_stddev = 0.05
    
        factor_stddev = torch.pow(
            self.target_stddev / torch.sqrt(torch.tensor(1.0 * self.max_rank)),
            1.0 / len(self.dims))
        self.factor_stddev = factor_stddev
        
        initializer_dist = TruncatedNormal(loc=0.0,scale=factor_stddev,a=-3.0*factor_stddev,b=3.0*factor_stddev)
        
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
        self.weights = None


    def _build_factor_distributions(self):

        factor_scale_multiplier = 1e-7

        factor_scales = [
            self.add_variable(factor_scale_multiplier *
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

        self.rank_parameter = self.add_variable(torch.sqrt(self.get_rank_parameters_update().clone().detach()).view([1,self.max_rank]),trainable=False)# Parameter(torch.sqrt(torch.tensor(self.get_rank_parameters_update())).view([1,self.max_rank]))

        self.factor_prior_distributions = []

        for x in self.dims:
            zero_mean = torch.zeros([x, self.max_rank])
            base_dist = td.Normal(loc=zero_mean,scale=self.rank_parameter)
            independent_dist = td.Independent(base_dist,reinterpreted_batch_ndims=2)
            self.factor_prior_distributions.append(independent_dist)#td.Independent(base_dist,reinterpreted_batch_ndims=2))

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


        with torch.no_grad():

            rank_update = self.get_rank_parameters_update()

            self.rank_parameter.data.sub_(self.rank_parameter.data)

            sqrt_parameter_update = torch.sqrt((1 - self.em_stepsize) * self.rank_parameter.data**2 + self.em_stepsize * rank_update)

            self.rank_parameter.data.add_(sqrt_parameter_update.to(self.rank_parameter.device))

    def get_rank(self, threshold=1e-4):
        return len(torch.where(self.get_rank_variance() > threshold))

    def get_kl_divergence_to_prior(self):

        kl_sum= 0.0

        for p in self.factor_distributions:
            var_ratio = (p.stddev / self.rank_parameter).pow(2)
            t1 = ((p.mean ) / self.rank_parameter).pow(2)
            kl = torch.sum(0.5 * (var_ratio + t1 - 1 - var_ratio.log()))
            kl_sum+=kl

        return kl_sum


class TensorTrain(LowRankTensor):
    def __init__(self, dims, max_rank, **kwargs):

        self.max_rank = max_rank

        if type(self.max_rank)==int:
            self.max_ranks = [1]+(len(dims)-1)*[self.max_rank]+[1]
        else:
            assert(type(max_rank)==list)
            self.max_ranks = max_rank
            self.max_rank = max(self.max_rank)


        super().__init__(dims, **kwargs)
        self.tensor_type = 'TensorTrain'

    def get_full(self):

        if hasattr(self,"masks"):
            raise NotImplementedError
            factors = [torch.multiply(x,y) for x,y in zip(self.factors,self.masks)]+[torch.multiply(torch.expand_dims(torch.expand_dims(self.masks[-1],axis=-1),axis=-1),self.factors[-1])]
            return tl.mps_to_tensor(factors)
        else:
            return tl.tt_to_tensor(self.factors)


    def estimate_rank(self, threshold=1e-4):

        return [int(sum(torch.square(x) > threshold)) for x in self.rank_parameters]


    def get_parameter_savings(self):
        
        rank_estimates = [1]+self.estimate_rank()+[1]

        reduced_rank_parameters = 0
        total_tt_parameters = sum([np.prod(x.shape) for x in self.factors])

        for i,x in enumerate(self.dims):
            reduced_rank_parameters+= rank_estimates[i]*x*rank_estimates[i+1]
            


        return total_tt_parameters-reduced_rank_parameters,np.prod(self.dims)-total_tt_parameters


    def _nn_init(self):

        if hasattr(self,"target_stddev"):
            pass
        else:
            self.target_stddev = 0.05

        factor_stddev = torch.pow(
            torch.pow(1.0 * self.max_rank, -self.order + 1) *
            torch.square(self.target_stddev), 1 / (2.0 * self.order))
        self.factor_stddev = factor_stddev

        sizes = [[1, self.dims[0], self.max_ranks[1]]] + [[self.max_ranks[i+1], x, self.max_ranks[i+2]] for i,x in enumerate(self.dims[1:-1])] + [[self.max_ranks[-2], self.dims[-1], 1]]

        initializer_dist = TruncatedNormal(loc=0.0,
                                               scale=factor_stddev,
                                               a=-self.order * factor_stddev,
                                               b=self.order * factor_stddev)
        init_factors = [initializer_dist.sample(x) for x in sizes]

        return init_factors

    def _random_init(self):

        factors = tl.random.random_mps(self.dims,
                                       self.max_ranks,
                                       full=False,
                                       random_state=getattr(
                                           self, 'seed', None))

        if hasattr(self, 'target_norm'):
            curr_norm = torch.norm(tl.mps_to_tensor(factors))
            multiplier = torch.pow(self.target_norm / curr_norm, 1 / self.order)
            factors = [multiplier * x for x in factors]

        return factors

    def _build_factors(self):

        if hasattr(self, 'initialization_method'):
            if self.initialization_method == 'random':
                self.factors = self._random_init()
            elif self.initialization_method == 'nn':
                self.factors = self._nn_init()
            elif self.initialization_method == 'svd':
                self.initialization_tensor = tf.reshape(self.initialization_tensor,self.dims)
                self.factors = tensorly.decomposition.matrix_product_state(
                    self.initialization_tensor, self.max_ranks)
            else:
                raise (ValueError("Initialization method not supported."))
        else:
            self.factors = self._random_init()

        self.factors = [self.add_variable(x) for x in self.factors]

        self.sizes = [x.shape for x in self.factors]

    def _build_factor_distributions(self):

        factor_scale_init = 1e-7

        factor_scales = [
            self.add_variable(factor_scale_init * torch.ones(factor.shape))
            for factor in self.factors
        ]

        self.factor_distributions = []

        for factor, factor_scale in zip(self.factors, factor_scales):
            self.factor_distributions.append(
                td.Independent(td.Normal(
                    loc=factor,
                    scale=factor_scale),
                                reinterpreted_batch_ndims=3))

    def _build_low_rank_prior(self):

        self.rank_parameters = [
            self.add_variable(torch.sqrt(x.clone().detach()))
            for x in self.get_rank_parameters_update()
        ]

        self.factor_prior_distributions = []

        for i in range(len(self.dims) - 1):

            self.factor_prior_distributions.append(
                td.Independent(td.Normal(
                    loc=torch.zeros(self.factors[i].shape),
                    scale=self.rank_parameters[i]),
                                reinterpreted_batch_ndims=3))

        self.factor_prior_distributions.append(
            td.Independent(td.Normal(
                loc=torch.zeros(self.factors[-1].shape),
                scale=self.rank_parameters[-1].unsqueeze(1).unsqueeze(2)),
                            reinterpreted_batch_ndims=3))

    def sample_full(self):
        
        factors = [x.rsample() for x in self.factor_distributions]

        if hasattr(self,"masks"):
            raise NotImplementedError
            factors = [tf.multiply(x,y) for x,y in zip(factors,self.masks)]+[tf.multiply(tf.expand_dims(tf.expand_dims(self.masks[-1],axis=-1),axis=-1),factors[-1])]
        
        return tl.tt_to_tensor(factors)
        

    def get_rank_parameters_update(self):

        updates = []
        
        for i in range(len(self.dims) - 1):

            M = torch.sum(torch.square(self.factor_distributions[i].mean) +
                              torch.square(self.factor_distributions[i].stddev),
                              axis=[0, 1])

            if i == len(self.dims) - 2:
                D = self.max_ranks[i] * self.dims[i] + self.dims[i + 1]
                M += torch.sum(
                    torch.square(self.factor_distributions[i + 1].mean) +
                    torch.square(self.factor_distributions[i + 1].stddev),
                    axis=[1, 2])
            else:
                D = self.max_ranks[i] * self.dims[i]

            if self.prior_type == 'log_uniform':
                update = M / (D + 1)

            elif self.prior_type == 'half_cauchy':
                update = (M - (self.eta**2) * D +
                          torch.sqrt(M**2 + (M * self.eta**2) * (2.0 * D + 8.0) +
                                  (D**2.0) * (self.eta**4.0))) / (2 * D + 4.0)

            updates.append(update)

        return updates

    def update_rank_parameters(self):

        with torch.no_grad():
            rank_updates = self.get_rank_parameters_update()

            for rank_parameter, rank_update in zip(self.rank_parameters, rank_updates):

                rank_parameter.data.sub_(rank_parameter.data)

                sqrt_parameter_update = torch.sqrt((1 - self.em_stepsize) * rank_parameter.data**2 + self.em_stepsize * rank_update)

                rank_parameter.data.add_(sqrt_parameter_update.to(rank_parameter.device))


    def get_rank(self, threshold=1e-5):
        return [int(sum(torch.square(x) > threshold)) for x in self.rank_parameters]

    def prune_ranks(self, threshold=1e-5):
        raise NotImplementedError
        self.masks =[tf.cast(tf.math.greater(x,threshold),tf.float32) for x in self.rank_parameters]

    def get_kl_divergence_to_prior(self):

        kl_sum= 0.0

        appended_rank_parameters = self.rank_parameters+[self.rank_parameters[-1].unsqueeze(1).unsqueeze(2)]

        for p,rank_parameter in zip(self.factor_distributions,appended_rank_parameters):
            var_ratio = (p.stddev / rank_parameter).pow(2)
            t1 = ((p.mean ) / rank_parameter).pow(2)
            kl = torch.sum(0.5 * (var_ratio + t1 - 1 - var_ratio.log()))
            kl_sum+=kl

        

        return kl_sum

#%%
dims = [50,50,50]
max_rank = 5
true_rank = 2
EM_STEPSIZE = 1.0

tensor = TensorTrain(dims=dims,max_rank=max_rank,prior_type='log_uniform',em_stepsize=EM_STEPSIZE)

full = tl.tt_to_tensor(tl.random.random_tt(shape=dims,rank=true_rank))


log_likelihood_dist = td.Normal(0.0,0.001)


def log_likelihood():
    return torch.mean(torch.stack([-torch.mean(log_likelihood_dist.log_prob(full-tensor.sample_full())) for _ in range(5)]))


def mse():
    return torch.norm(full-tensor.get_full())/torch.norm(full)

def kl_loss():
    return log_likelihood()+tensor.get_kl_divergence_to_prior()


loss = kl_loss

#loss = log_likelihood

optimizer = torch.optim.Adam(tensor.trainable_variables,lr=1e-5)

#%%

for i in range(10000):

    optimizer.zero_grad()

    loss_value = loss()

    loss_value.backward()

    optimizer.step()

    tensor.update_rank_parameters()

    if i%1000==0:
        print('Loss ',loss())
        print('RMSE ',mse())
        print('Rank ',tensor.estimate_rank())
        print(tensor.rank_parameters)


#%%

print(tensor.factor_prior_distributions[-1].stddev[:,0,0])
print(tensor.rank_parameters[1])


#%%
tensor.update_rank_parameters()