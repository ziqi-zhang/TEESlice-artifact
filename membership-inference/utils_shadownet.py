import torch

# choose w as 2D array
def shadownet_layer(ori_model, trained_model, kernel_name, r, t, use_shuffle=True):
    trained_layer = trained_model.state_dict()[kernel_name]
    kernel_size = trained_layer.size()[1] * trained_layer.size()[2] * trained_layer.size()[3]
    ori_n_outchannels = trained_layer.size()[0]
    trained_layer_flatten = torch.zeros(ori_n_outchannels, kernel_size)
    for ki in range(ori_n_outchannels):
        trained_layer_flatten[ki,:] = torch.flatten(trained_layer[ki,:,:,:])
    ori_layer = ori_model.state_dict()[kernel_name]
    ori_layer_flatten = torch.zeros(ori_n_outchannels, kernel_size)
    for ki in range(ori_n_outchannels):
        ori_layer_flatten[ki,:] = torch.flatten(ori_layer[ki,:,:,:])

    # Generate random f array
    assert(r>=1)
    new_n_outchannels = int(r * ori_n_outchannels)
    f_cnt = new_n_outchannels - ori_n_outchannels
    if f_cnt <= 0:
        print(kernel_name, trained_layer.shape)
        new_n_outchannels = ori_n_outchannels + 1
        f_cnt = 1
        # set_trace()
    f_array = torch.FloatTensor(f_cnt, kernel_size).uniform_(-t, t)
    # print(f_array, f_array.shape)

    # Generate random indices for f
    frand_indices = torch.randint(low=0, high=f_cnt, size=(ori_n_outchannels,))
    # print("f rand indices: ", frand_indices)
    new_layer_flatten = torch.zeros(new_n_outchannels, kernel_size)
    
    # Add f
    for ki in range(ori_n_outchannels):
        new_layer_flatten[ki,:] = trained_layer_flatten[ki,:] + \
            f_array[frand_indices[ki], :]
    new_layer_flatten[ori_n_outchannels:, :] = f_array

    # Shuffle weights and f
    if use_shuffle:
        shuffle_indices = torch.randperm(ori_n_outchannels)
    else:
        shuffle_indices = torch.arange(ori_n_outchannels)
    # print("Shuffle indices: ", shuffle_indices)
    new_layer_flatten[:ori_n_outchannels, :] = new_layer_flatten[shuffle_indices, :]
    frand_indices = frand_indices[shuffle_indices]
    # print("Shuffle f rand indices: ", frand_indices)
    
    return ori_layer_flatten, trained_layer_flatten, new_layer_flatten, \
        frand_indices, shuffle_indices


# Return True for correct f; False for incorrect f
def fchecker_variance(cur_flatten, t, ratio=0.01):
    cur_var = torch.var(cur_flatten)
    uniform_var = 16*t*t / 12
    if cur_var < ratio * uniform_var:
        return True
    return False

# Use L2 norm as the distance
def Lx_distance(ori_flatten, cur_flatten, dist_norm=2):
    return torch.norm(ori_flatten-cur_flatten, p=dist_norm)


def attack_shadownet_layer_nolamb(ori_layer_flatten, trained_layer_flatten, new_layer_flatten, \
    r, t, frand_indices, shuffle_indices):
    ori_n_outchannels = trained_layer_flatten.shape[0]
    new_n_outchannels = new_layer_flatten.shape[0]
    f_cnt = new_n_outchannels - ori_n_outchannels
    dist_norm = 2

    # Get f array
    f_array = new_layer_flatten[ori_n_outchannels:, :]

    # Enumerate to find the correct f
    fguess_indices = []
    guess_trained_layer = torch.zeros_like(trained_layer_flatten)
    for ki in range(ori_n_outchannels):
        for fi in range(f_cnt):
            minus_flatten = new_layer_flatten[ki,:] - f_array[fi,:]
            is_correct = fchecker_variance(minus_flatten, t)
            if is_correct:
                fguess_indices.append(fi)
                guess_trained_layer[ki, :] = minus_flatten
                # Check incorrect results
                if fi != frand_indices[ki]:
                    print("cur minus: ", minus_flatten[:8])
                    print("true minus: ", (new_layer_flatten[ki,:] - f_array[frand_indices[ki], :])[:8] )
                    print("false f: ", f_array[fi,:][:8])
                    print("true f: ", f_array[frand_indices[ki], :][:8])
                    print("===================")
                break
    # print("f guess indices: ", fguess_indices)

    shuffle_guess_indices = torch.zeros_like(shuffle_indices)
    # Enumerate to find the right shuffle order with Lx norm
    for ki in range(ori_n_outchannels):
        min_dist = -1
        cur_flatten = guess_trained_layer[ki,:]
        for oki in range(ori_n_outchannels):
            cur_dist = Lx_distance(ori_layer_flatten[oki,:], cur_flatten, dist_norm)
            if (min_dist < 0) or min_dist > cur_dist:
                used_oki_flag = False
                # for pki in range(ki):
                #     if shuffle_guess_indices[pki] == oki:
                #         used_oki_flag = True
                #         break
                if not used_oki_flag:
                    min_dist = cur_dist
                    shuffle_guess_indices[ki] = oki

    # print("shuffle guess indices: ", shuffle_guess_indices)
    equal_arr = (shuffle_indices == shuffle_guess_indices)
    print(equal_arr.sum().item(), '/', ori_n_outchannels)

    return shuffle_indices, shuffle_guess_indices