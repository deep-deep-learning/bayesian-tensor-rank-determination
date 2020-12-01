import numpy as np


dims = [[125,220,250],[2,2,4]]

shape0 = [[200,220,250],[125,130,136],[200,200,209],[166,175,188],[200,200,200]]
shape1 = [4,4,8]


baseline_params = [10131227,2202608,8351593,5461306,7046547]
baseline_params = [128*x for x in baseline_params] 
print(baseline_params)

max_ranks = {'CP':[350,306,333,326,335],'TensorTrainMatrix':[16,16,16,16,16],'TensorTrain':[24,24,24,24,24],'Tucker':[22,20,22,21,22]}

true_ranks = {'CP':[350,306,333,326,335],'TensorTrainMatrix':[16,16,16,16,16],'TensorTrain':[24,24,24,24,24],'Tucker':[22,20,22,21,22]}

tensorized_params = {}
ard_params = {}
for key in ['CP','TensorTrainMatrix','TensorTrain','Tucker']:
    tensorized_params[key]=[]
    ard_params[key]=[]


ii = 0

dims = [shape0[ii],shape1]

ttm_rank = 16
cp_rank = 335
tt_rank = 25
tucker_rank = 22
"""
def get_ttm_params(shape,rank)

order=len(dims[0])
ranks = [[1,ttm_rank]]+[[ttm_rank,ttm_rank] for _ in range(order-2)]+[[ttm_rank,1]]
dim_pairs = [[x,y] for x,y in zip(dims[0],dims[1])]
ttm_params = sum([np.prod(x+y) for x,y in zip(dim_pairs,ranks)])
print(dim_pairs,ranks)
print("TTM_params", ttm_params)
"""

def get_tt_params(tt_dims,tt_rank):

    if type(tt_rank)!=list:
        tt_rank = [1]+(len(tt_dims)-1)*[tt_rank]+[1]

    print(tt_rank,tt_dims)
    order = len(tt_dims)
    ranks = [[tt_rank[i],tt_rank[i+1]] for i in range(order)]
    print(ranks)
    tt_params = sum([np.prod([x]+y) for x,y in zip(tt_dims,ranks)])
    return tt_params
    
tt_dims = dims[0]+[np.prod(dims[1])]
tt_rank = max_ranks['TensorTrain'][3]
 

print("TT_params ",get_tt_params(tt_dims,tt_rank))


"""

def get_cp_params(shape,rank)

cp_params = 0.0
while cp_params<ttm_params:
    cp_dims = dims[0]+[np.prod(dims[1])]
    cp_params = cp_rank*sum(cp_dims)
    print("CP_params ",cp_params," cp rank ",cp_rank)
    cp_rank+=1

def get_tucker_params(shape,rank)

tucker_params = 0.0
while tucker_params<ttm_params:
    tucker_dims = dims[0]+[np.prod(dims[1])]
    tucker_params = (tucker_rank**len(tucker_dims))+tucker_rank*sum(tucker_dims)
    print("Tucker params ",tucker_params," tucker rank ",tucker_rank)
    tucker_rank+=1

print("CP ",cp_rank,"\nTTM ",ttm_rank,"\nTT ",tt_rank,"\nTucker ",tucker_rank)"""


print("baseline parameters",sum(baseline_params))