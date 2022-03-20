# This script contains the python implementation of FPLinQ algorithm (original work by K. Shen and W. Yu)
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
import helper_functions

# Parallel computation over multiple layouts
def FP_optimize(general_para, g, weights):
    number_of_samples, N, _ = np.shape(g)
    assert np.shape(g)==(number_of_samples, N, N)
    assert np.shape(weights)==(number_of_samples, N)
    g_diag = helper_functions.get_directLink_channel_losses(g)
    g_nondiag = helper_functions.get_crossLink_channel_losses(g)
    # For matrix multiplication and dimension matching requirement, reshape into column vectors
    weights = np.expand_dims(weights, axis=-1)
    g_diag = np.expand_dims(g_diag, axis=-1)
    x = np.ones([number_of_samples, N, 1])
    tx_power = general_para.tx_power
    output_noise_power = general_para.output_noise_power
    tx_powers = np.ones([number_of_samples, N, 1]) * tx_power  # assume same power for each transmitter
    # Run 100 iterations of FP computations
    # In the computation below, every step's output is with shape: number of samples X N X 1
    for i in range(100):
        # Compute z
        p_x_prod = x * tx_powers
        z_denominator = np.matmul(g_nondiag, p_x_prod) + output_noise_power
        z_numerator = g_diag * p_x_prod
        z = z_numerator / z_denominator
        # compute y
        y_denominator = np.matmul(g, p_x_prod) + output_noise_power
        y_numerator = np.sqrt(z_numerator * weights * (z + 1))
        y = y_numerator / y_denominator
        # compute x
        x_denominator = np.matmul(np.transpose(g, (0,2,1)), np.power(y, 2)) * tx_powers
        x_numerator = y * np.sqrt(weights * (z + 1) * g_diag * tx_powers)
        x_new = np.power(x_numerator / x_denominator, 2)
        x_new[x_new > 1] = 1  # thresholding at upperbound 1
        x = x_new
    assert np.shape(x)==(number_of_samples, N, 1)
    x_final = np.squeeze(x, axis=-1)
    return x_final

def FP(weights, g, var_noise, input_x):
    number_of_samples, N, _ = np.shape(g)
    assert np.shape(g)==(number_of_samples, N, N)
    assert np.shape(weights)==(number_of_samples, N)
    g_diag = helper_functions.get_directLink_channel_losses(g)
    g_nondiag = helper_functions.get_crossLink_channel_losses(g)
    # For matrix multiplication and dimension matching requirement, reshape into column vectors
    weights = np.expand_dims(weights, axis=-1)
    g_diag = np.expand_dims(g_diag, axis=-1)
    x = input_x#np.ones([number_of_samples, N, 1])
    #tx_power = general_para.tx_power
    #output_noise_power = general_para.output_noise_power
    # tx_powers = np.ones([number_of_samples, N, 1]) * tx_power  # assume same power for each transmitter
    # Run 100 iterations of FP computations
    # In the computation below, every step's output is with shape: number of samples X N X 1
    for i in range(100):
        # Compute z
        p_x_prod = x 
        z_denominator = np.matmul(g_nondiag, p_x_prod) + var_noise
        z_numerator = g_diag * p_x_prod
        z = z_numerator / z_denominator
        # compute y
        y_denominator = np.matmul(g, p_x_prod) + var_noise
        y_numerator = np.sqrt(z_numerator * weights * (z + 1))
        y = y_numerator / y_denominator
        # compute x
        x_denominator = np.matmul(np.transpose(g, (0,2,1)), np.power(y, 2))
        x_numerator = y * np.sqrt(weights * (z + 1) * g_diag)
        x_new = np.power(x_numerator / x_denominator, 2)
        x_new[x_new > 1] = 1  # thresholding at upperbound 1
        x = x_new
    assert np.shape(x)==(number_of_samples, N, 1)
    x_final = np.squeeze(x, axis=-1)
    return x_final
