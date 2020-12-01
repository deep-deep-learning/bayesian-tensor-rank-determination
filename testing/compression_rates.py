#%%
import numpy as np


def get_tucker_params(tucker_dims, tucker_rank):

    if type(tucker_rank) != list:
        tucker_rank = len(tucker_dims) * [tucker_rank]

    tucker_dims = dims[0] + [np.prod(dims[1])]
    tucker_params = np.prod(tucker_rank) + sum(
        [x * y for x, y in zip(tucker_rank, tucker_dims)])

    return tucker_params


def get_cp_params(cp_dims, cp_rank):

    cp_params = cp_rank * sum(cp_dims)
    return cp_params


def get_ttm_params(ttm_dims, ttm_rank):

    if type(ttm_rank) != list:
        ttm_rank = [1] + (len(ttm_dims[0]) - 1) * [ttm_rank] + [1]
    print(ttm_rank)
    print(ttm_dims)
    order = len(ttm_dims[0])
    ranks = [[ttm_rank[i], ttm_rank[i + 1]] for i in range(order)]
    print(ranks)
    ttm_params = sum(
        [np.prod(x) * np.prod(y) for x, y in zip(zip(*ttm_dims), ranks)])
    return ttm_params


def get_tt_params(tt_dims, tt_rank):

    if type(tt_rank) != list:
        tt_rank = [1] + (len(tt_dims) - 1) * [tt_rank] + [1]

    order = len(tt_dims)
    ranks = [[tt_rank[i], tt_rank[i + 1]] for i in range(order)]
    tt_params = sum([np.prod([x] + y) for x, y in zip(tt_dims, ranks)])
    return tt_params


def get_params_wrapper(tensor_type, dims, rank):
    if tensor_type == 'Tucker':
        return get_tucker_params(dims, rank)
    elif tensor_type == 'CP':
        return get_cp_params(dims, rank)
    elif tensor_type == 'TensorTrain':
        return get_tt_params(dims, rank)
    elif tensor_type == 'TensorTrainMatrix':
        return get_ttm_params(dims, rank)


shape0 = [[200, 220, 250], [125, 130, 136], [200, 200, 209], [166, 175, 188],
          [200, 200, 200]]
shape1 = [4, 4, 8]

baseline_params = [10131227, 2202608, 8351593, 5461306, 7046547]
baseline_params = [128 * x for x in baseline_params]
print(baseline_params)

max_ranks = {
    'CP': [350, 306, 333, 326, 335],
    'TensorTrainMatrix': [16, 16, 16, 16, 16],
    'TensorTrain': [24, 24, 24, 24, 24],
    'Tucker': [22, 20, 22, 21, 22]
}

true_ranks = {
    'CP': [350, 306, 333, 326, 335],
    'TensorTrainMatrix': [16, 16, 16, 16, 16],
    'TensorTrain': [24, 24, 24, 24, 24],
    'Tucker': [22, 20, 22, 21, 22]
}

tensorized_params = {}
ard_params = {}
for key in ['CP', 'TensorTrainMatrix', 'TensorTrain', 'Tucker']:
    tensorized_params[key] = []
    ard_params[key] = []

ii = 0

dims = [shape0[ii], shape1]

ttm_dims = dims
ttm_rank = max_ranks['TensorTrainMatrix'][ii]
ttm_params = get_ttm_params(ttm_dims, ttm_rank)
print("TTM params", ttm_params)

tt_dims = dims[0] + [np.prod(dims[1])]
tt_rank = max_ranks['TensorTrain'][ii]
tt_params = get_tt_params(tt_dims, tt_rank)
print("TT params", tt_params)

cp_dims = tt_dims
cp_rank = max_ranks['CP'][ii]
cp_params = get_cp_params(cp_dims, cp_rank)
print("CP_params ", cp_params)

print("baseline parameters", sum(baseline_params))
tucker_dims = tt_dims
tucker_rank = max_ranks['Tucker'][ii]
tucker_params = get_tucker_params(tucker_dims, tucker_rank)
print("Tucker_params ", tucker_params)

print("Tucker_params ", get_params_wrapper('Tucker', tucker_dims, tucker_rank))

#%%
"""
for ii in range(len(shape0)):

    ttm_dims = dims
    ttm_rank = max_ranks['TensorTrainMatrix'][ii]
    ttm_params = get_ttm_params(ttm_dims,ttm_rank) 
    print("TTM params",ttm_params)

        
    tt_dims = dims[0]+[np.prod(dims[1])]
    tt_rank = max_ranks['TensorTrain'][ii]
    tt_params = get_tt_params(tt_dims,tt_rank) 
    print("TT params",tt_params)


    cp_dims = tt_dims
    cp_rank = max_ranks['CP'][ii]
    cp_params = get_cp_params(cp_dims,cp_rank) 
    print("CP_params ",cp_params)


    tucker_dims = tt_dims
    tucker_rank = max_ranks['Tucker'][ii]
    tucker_params = get_tucker_params(tucker_dims,tucker_rank) 
    print("Tucker_params ",tucker_params)

    tensorized_params = 
    ard_params = {}
for key in ['CP','TensorTrainMatrix','TensorTrain','Tucker']:
    tensorized_params[key]=[]
    ard_params[key]=[]

"""
