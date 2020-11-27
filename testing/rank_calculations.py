import numpy as np


dims = [[125,220,250],[2,2,4]]

shape0 = [[200,220,250],[125,130,136],[200,200,209],[166,175,188],[200,200,200]]
shape1 = [2,2,4]

ii = 0

dims = shape0[ii]



ttm_rank = 20
order=len(dims[0])
ranks = [[1,ttm_rank]]+[[ttm_rank,ttm_rank] for _ in range(order-2)]+[[ttm_rank,1]]
dim_pairs = [[x,y] for x,y in zip(dims[0],dims[1])]
ttm_params = ttm_rank*200*2 + ttm_rank*220*2*ttm_rank + ttm_rank*250*4
print("TTM_params ",ttm_params)
new_ttm_params = sum([np.prod(x+y) for x,y in zip(dim_pairs,ranks)])
print(new_ttm_params)



tt_rank = 21
tt_dims = dims[0]+[np.prod(dims[1])]
order = len(tt_dims)
ranks = [[1,tt_rank]]+[[tt_rank,tt_rank] for _ in range(order-2)]+[[tt_rank,1]]
tt_params = sum([np.prod([x]+y) for x,y in zip(tt_dims,ranks)])
print("TT_params ",tt_params)


cp_dims = dims[0]+[np.prod(dims[1])]
cp_rank = 300
cp_params = cp_rank*sum(cp_dims)
print("CP_params ",cp_params)


tucker_rank = 21
tucker_dims = dims[0]+[np.prod(dims[1])]
tucker_params = (tucker_rank**len(tucker_dims))+tucker_rank*sum(tucker_dims)
print("Tucker params ",tucker_params)
