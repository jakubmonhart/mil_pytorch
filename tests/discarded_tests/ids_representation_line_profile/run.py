import torch

@profile
def forward(input, prepNN, afterNN, aggregation_func):    
    ids = input[1]
    input = input[0]
    NN_out = prepNN(input) # Forward all instances through neural network
    # print(NN_out)
    
    inner_ids = ids[len(ids)-1]
    # print(inner_ids)
    # Numpy version of this segment is faster on CPU (cca 2x - 3x faster .. the differenec is more significatn for longer arrays)
    # start = time.time()
    unique, inverse, counts = torch.unique(inner_ids, sorted = True, return_inverse = True, return_counts = True)
    idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
    bags = unique[idx]
    counts = counts[idx]
    # print('Unique | Elapsed time: {}'.format(time.time()-start))

    # Allocate memory for output
    # start = time.time()
    output = torch.empty((len(bags), len(NN_out[0])))

    for i, bag in enumerate(bags):
        output[i] = aggregation_func(NN_out[inner_ids == bag], dim = 0)
    
    output = afterNN(output.double())
    # print('Aggregation + afterNN | Elapsed time: {}'.format(time.time()-start))


    if (ids.shape[0] == 1):
        return output
    else:
        # start = time.time()
        ids = ids[:len(ids)-1]
        mask = torch.empty(0).long()
        for i in range(len(counts)):
            mask = torch.cat((mask, torch.sum(counts[:i], dtype = torch.int64).reshape(1)))
        # print('Mask for ids | Elapsed time: {}'.format(time.time() - start))
        return (output, ids[:,mask])


# Pre and after agg function
prepNN1 = torch.nn.Sequential(
    torch.nn.Linear(50, 5, bias = True),
    torch.nn.ReLU(),
).double()

afterNN1 = torch.nn.Sequential(
    torch.nn.Linear(5, 1, bias = True),
    torch.nn.Tanh()
).double()

#Data
data = torch.rand((10, 50)).double()
ids = torch.Tensor([[1, 1, 1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 2, 2, 3, 3, 3, 4]]).double()

output = forward((data,ids), prepNN1, afterNN1, torch.mean)

print(output)