# This script contains all the helper functions
# for the work "Spatial Deep Learning for Wireless Scheduling",
# available at https://ieeexplore.ieee.org/document/8664604.

# For any reproduce, further research or development, please kindly cite our JSAC journal paper:
# @Article{spatial_learn,
#    author = "W. Cui and K. Shen and W. Yu",
#    title = "Spatial Deep Learning for Wireless Scheduling",
#    journal = "{\it IEEE J. Sel. Areas Commun.}",
#    year = 2019,
#    volume = 37,
#    issue = 6,
#    pages = "1248-1261",
#    month = "June",
# }


import numpy as np

# Generate layout one at a time
def layout_generate(general_para):
    N = general_para.n_links
    # first, generate transmitters' coordinates
    tx_xs = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    tx_ys = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    while(True): # loop until a valid layout generated
        # generate rx one by one rather than N together to ensure checking validity one by one
        rx_xs = []; rx_ys = []
        for i in range(N):
            got_valid_rx = False
            while(not got_valid_rx):
                pair_dist = np.random.uniform(low=general_para.shortest_directLink_length, high=general_para.longest_directLink_length)
                pair_angles = np.random.uniform(low=0, high=np.pi*2)
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)
                if(0<=rx_x<=general_para.field_length and 0<=rx_y<=general_para.field_length):
                    got_valid_rx = True
            rx_xs.append(rx_x); rx_ys.append(rx_y)
        # For now, assuming equal weights and equal power, so not generating them
        layout = np.concatenate((tx_xs, tx_ys, rx_xs, rx_ys), axis=1)
        distances = np.zeros([N, N])
        # compute distance between every possible Tx/Rx pair
        for rx_index in range(N):
            for tx_index in range(N):
                tx_coor = layout[tx_index][0:2]
                rx_coor = layout[rx_index][2:4]
                # according to paper notation convention, Hij is from jth transmitter to ith receiver
                distances[rx_index][tx_index] = np.linalg.norm(tx_coor - rx_coor)
        # Check whether a tx-rx link (potentially cross-link) is too close
        if(np.min(distances)>general_para.shortest_crossLink_length):
                 break
    return layout, distances

# Input: allocs: layouts X N; directlink_channel_losses: layouts X N; crosslink_channel_losses: layouts X N X N
# Output: SINRs: layouts X N
def compute_SINRs(var_noise, allocs, directlink_channel_losses, crosslink_channel_losses):
    assert np.shape(directlink_channel_losses) == np.shape(allocs), \
        "Mismatch shapes: {} VS {}".format(np.shape(directlink_channel_losses), np.shape(allocs))
    SINRs_numerators = allocs * directlink_channel_losses  # layouts X N
    SINRs_denominators = np.squeeze(np.matmul(crosslink_channel_losses, np.expand_dims(allocs, axis=-1))) + var_noise  # layouts X N
    SINRs = SINRs_numerators / SINRs_denominators  # layouts X N
    return SINRs

# Input: allocs: layouts X N; directlink_channel_losses: layouts X N; crosslink_channel_losses: layouts X N X N
# Output: rates: layouts X N
def compute_rates(var_noise, allocs, directlink_channel_losses, crosslink_channel_losses):
    SINRs = compute_SINRs(var_noise, allocs, directlink_channel_losses, crosslink_channel_losses)
    rates = np.log2(1 + SINRs) # layouts X N general_para.bandwidth * 
#
    return rates

def get_directLink_channel_losses(channel_losses):
    return np.diagonal(channel_losses, axis1=1, axis2=2)  # layouts X N

def get_crossLink_channel_losses(channel_losses):
    N = np.shape(channel_losses)[-1]
    return channel_losses * ((np.identity(N) < 1).astype(float))

# Add in shadowing into channel losses
def add_shadowing(channel_losses):
    shadow_coefficients = np.random.normal(loc=0, scale=8, size=np.shape(channel_losses))
    channel_losses = channel_losses * np.power(10.0, shadow_coefficients / 10)
    return channel_losses

# Add in fast fading into channel nosses
def add_fast_fading(channel_losses):
    fastfadings = (np.power(np.random.normal(loc=0, scale=1, size=np.shape(channel_losses)), 2) +
                   np.power(np.random.normal(loc=0, scale=1, size=np.shape(channel_losses)), 2)) / 2
    channel_losses = channel_losses * fastfadings
    return channel_losses

def update_proportional_fairness_weights(weights, rates):
    alpha = 0.95
    return 1 / (alpha / weights + (1 - alpha) * rates)

# Find binary approximation of importance weights parallelly over multiple layouts
def binarize_proportional_fairness_weights(general_para, weights):
    N = general_para.n_links
    n_layouts = np.shape(weights)[0]
    assert np.shape(weights) == (n_layouts, N)
    sorted_indices = np.argsort(weights, axis=1)
    weights_normalized = weights / np.linalg.norm(weights,axis=1,keepdims=True) # normalize to l2 norm 1
    # initialize variables
    binary_weights = np.zeros([n_layouts, N])
    max_dot_product = np.zeros(n_layouts)
    # use greedy to activate one at a time
    for i in range(N-1, -1, -1):
        binary_weights[np.arange(n_layouts), sorted_indices[:,i]] = 1
        binary_weights_normalized = binary_weights/np.linalg.norm(binary_weights,axis=1,keepdims=True)
        current_dot_product = np.einsum('ij,ij->i', weights_normalized, binary_weights_normalized)
        binary_weights[np.arange(n_layouts), sorted_indices[:,i]] = (current_dot_product >= max_dot_product).astype(int)
        max_dot_product = np.maximum(max_dot_product, current_dot_product)
    return binary_weights

def get_pair_indices(general_para, locations):
    N = general_para.n_links
    assert np.shape(locations)==(N,2), "[get_pair_indices] input locations argument with wrong shape: {}".format(np.shape(locations))
    locations_indices = np.floor(locations / np.array([general_para.cell_length, general_para.cell_length])).astype(int)
    # deal with stations located at upper boundaries
    locations_indices[locations_indices[:,0] >= general_para.n_grids, 0] = general_para.n_grids - 1
    locations_indices[locations_indices[:,1] >= general_para.n_grids, 1] = general_para.n_grids - 1
    assert np.shape(locations_indices)==(N,2), "[get_pair_indices] generated location indices with wrong shape: {}".format(locations_indices)
    return locations_indices

def append_indices_array(general_para, indices):
    N = general_para.n_links
    n_layouts = np.shape(indices)[0]
    # Firstly, append at the end the indices that each link appears in the layout, parallelly over all layouts
    pad = np.tile(np.expand_dims(np.expand_dims(np.arange(n_layouts), -1), -1), [1, N, 1])
    indices_extract = np.concatenate([pad,indices], axis=2)
    assert np.shape(indices_extract) == (n_layouts, N, 3), "Wrong shape: {}".format(np.shape(indices_extract))
    # Secondly, append at the end the hash distinctive indices for summation among all links later
    hash_step = general_para.n_grids + 1
    hash_indices = np.dot(indices_extract, np.array([[hash_step**2], [hash_step],[1]]))
    indices_hash = np.concatenate([indices_extract, hash_indices], axis=-1)
    indices_hash = np.reshape(indices_hash, [n_layouts*N, 4])
    return indices_extract, indices_hash

# Processing test layout inputs into inputs for the conv net model
def process_layouts_inputs(general_para, layouts):
    N = general_para.n_links
    n_layouts = np.shape(layouts)[0]
    assert np.shape(layouts) == (n_layouts, N, 4)
    data_dict = dict()
    data_dict['layouts'] = layouts
    data_dict['pair_dists'] = np.linalg.norm(layouts[:, :, 0:2] - layouts[:, :, 2:4], axis=2)
    # compute station indices within the grid
    tx_indices = []
    rx_indices = []
    for i in range(n_layouts):
        txs = layouts[i, :, 0:2]
        rxs = layouts[i, :, 2:4]
        tx_indices.append(get_pair_indices(general_para, txs))
        rx_indices.append(get_pair_indices(general_para, rxs))
    data_dict['tx_indices'] = np.array(tx_indices)
    data_dict['rx_indices'] = np.array(rx_indices)
    # Assuming the convolutional filter size is 63X63
    conv_filter_center = np.floor(np.array([63, 63]) / 2).astype(int)
    data_dict['pair_tx_convfilter_indices'] = data_dict['tx_indices'] - data_dict['rx_indices'] + conv_filter_center
    data_dict['pair_rx_convfilter_indices'] = data_dict['rx_indices'] - data_dict['tx_indices'] + conv_filter_center
    # Add additional indices required by the neural network model to each tx/rx grid index
    data_dict['tx_indices_ext'], data_dict['tx_indices_hash'] = append_indices_array(general_para, data_dict['tx_indices'])
    data_dict['rx_indices_ext'], data_dict['rx_indices_hash'] = append_indices_array(general_para, data_dict['rx_indices'])
    return data_dict

def visualize_layout(ax, layout):
    N = np.shape(layout)[0]
    assert np.shape(layout) == (N, 4)
    tx_locs = layout[:, 0:2]
    rx_locs = layout[:, 2:4]
    ax.scatter(tx_locs[:, 0], tx_locs[:, 1], c='r', marker='x', label='Tx', s=9)
    ax.scatter(rx_locs[:, 0], rx_locs[:, 1], c='b', label='Rx', s=9)
    ax.legend()
    return

def visualize_superSet_on_layout(ax, layout, superset):
    N = np.shape(layout)[0]
    assert np.shape(layout) == (N, 4)
    assert np.size(superset) == N
    tx_locs = layout[:, 0:2]
    rx_locs = layout[:, 2:4]
    for i in range(N):  # label all links within the selected superset
        if(superset[i]!=0):
            ax.plot([tx_locs[i, 0], rx_locs[i, 0]], [tx_locs[i, 1], rx_locs[i, 1]], c='r', linewidth=4.0, alpha=0.3)
    return

def visualize_schedules_on_layout(ax, layout, schedules):
    N = np.shape(layout)[0]
    assert np.shape(layout) == (N, 4)
    assert np.size(schedules) == N
    tx_locs = layout[:, 0:2]
    rx_locs = layout[:, 2:4]
    for i in range(N):  # plot all activated links
        if(schedules[i] != 0):
            ax.plot([tx_locs[i, 0], rx_locs[i, 0]], [tx_locs[i, 1], rx_locs[i, 1]], c='k')
    return
