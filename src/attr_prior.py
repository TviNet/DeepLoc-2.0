import scipy
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage
try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))

    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)

    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)
        
def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)



# Smoothing amount for gradients before computing attribution prior loss;
# Smoothing window size is 1 + (2 * sigma); set to 0 for no smoothing
att_prior_grad_smooth_sigma = 3

# Maximum frequency integer to consider for a Fourier attribution prior
fourier_att_prior_freq_limit = 100

# Amount to soften the Fourier attribution prior loss limit; set to None
# to not soften; softness decays like 1 / (1 + x^c) after the limit
fourier_att_prior_freq_limit_softness = None
def place_tensor(tensor, input_tensor):
    """
    Places a tensor on GPU, if PyTorch sees CUDA; otherwise, the returned tensor
    remains on CPU.
    """
    return tensor.to(input_tensor.device)
        
def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=truncate)
    kernel = torch.tensor(kernel, dtype=torch.float32, device=input_tensor.device)

    # Expand the input and kernel to 3D, with channels of 1
    # Also make the kernel float-type, as the input is going to be of type float
    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1).float()
    padded_input = F.pad(input_tensor, (sigma,sigma),"replicate")
    smoothed = torch.nn.functional.conv1d(
        padded_input, kernel
    )
    return torch.squeeze(smoothed, dim=1)


def fourier_att_prior_loss_dct(
        input_grads, freq_limit, limit_softness,
        att_prior_grad_smooth_sigma
    ):
    """
    Computes an attribution prior loss for some given training examples,
    using a Fourier transform form.
    Arguments:
        `status`: a B-tensor, where B is the batch size; each entry is 1 if
            that example is to be treated as a positive example, and 0
            otherwise
        `input_grads`: a B x L x D tensor, where B is the batch size, L is
            the length of the input, and D is the dimensionality of each
            input base; this needs to be the gradients of the input with
            respect to the output (for multiple tasks, this gradient needs
            to be aggregated); this should be *gradient times input*
        `freq_limit`: the maximum integer frequency index, k, to consider for
            the loss; this corresponds to a frequency cut-off of pi * k / L;
            k should be less than L / 2
        `limit_softness`: amount to soften the limit by, using a hill
            function; None means no softness
        `att_prior_grad_smooth_sigma`: amount to smooth the gradient before
            computing the loss
    Returns a single scalar Tensor consisting of the attribution loss for
    the batch.
    """
    abs_grads = torch.abs(input_grads)

    # Smooth the gradients
    grads_smooth = abs_grads 
    # smooth_tensor_1d(
    #     abs_grads, att_prior_grad_smooth_sigma
    # )

    # Only do the positives
    pos_grads = grads_smooth

    # Loss for positives
    if pos_grads.nelement():
        #pos_fft = torch.rfft(pos_grads.float(), 1)
        pos_dct = dct(pos_grads.float(), "ortho")
        pos_mags = torch.abs(pos_dct)
        pos_mag_sum = torch.sum(pos_mags, dim=1, keepdim=True)
        pos_mag_sum[pos_mag_sum == 0] = 1  # Keep 0s when the sum is 0
        pos_mags = pos_mags / pos_mag_sum

        # Cut off DC
        pos_mags = pos_mags[:, 1:]

        # Construct weight vector
        weights = place_tensor(torch.ones_like(pos_mags), input_grads)
        if limit_softness is None:
            weights[:, freq_limit:] = 0
        else:
            x = place_tensor(
                torch.arange(1, pos_mags.size(1) - freq_limit + 1), input_grads
            ).float()
            weights[:, freq_limit:] = 1 / (1 + torch.pow(x, limit_softness))

        # Multiply frequency magnitudes by weights
        pos_weighted_mags = pos_mags * weights

        # Add up along frequency axis to get score
        pos_score = torch.sum(pos_weighted_mags, dim=1)
        pos_loss = 1 - pos_score
        return torch.mean(pos_loss)
    else:
        return place_tensor(torch.zeros(1), input_grads)


def fourier_att_prior_loss(
        input_grads, freq_limit, limit_softness,
        att_prior_grad_smooth_sigma
    ):
    """
    Computes an attribution prior loss for some given training examples,
    using a Fourier transform form.
    Arguments:
        `status`: a B-tensor, where B is the batch size; each entry is 1 if
            that example is to be treated as a positive example, and 0
            otherwise
        `input_grads`: a B x L x D tensor, where B is the batch size, L is
            the length of the input, and D is the dimensionality of each
            input base; this needs to be the gradients of the input with
            respect to the output (for multiple tasks, this gradient needs
            to be aggregated); this should be *gradient times input*
        `freq_limit`: the maximum integer frequency index, k, to consider for
            the loss; this corresponds to a frequency cut-off of pi * k / L;
            k should be less than L / 2
        `limit_softness`: amount to soften the limit by, using a hill
            function; None means no softness
        `att_prior_grad_smooth_sigma`: amount to smooth the gradient before
            computing the loss
    Returns a single scalar Tensor consisting of the attribution loss for
    the batch.
    """
    abs_grads = torch.abs(input_grads)

    # Smooth the gradients
    grads_smooth = abs_grads 
    # smooth_tensor_1d(
    #     abs_grads, att_prior_grad_smooth_sigma
    # )

    # Only do the positives
    pos_grads = grads_smooth

    # Loss for positives
    if pos_grads.nelement():
        #pos_fft = torch.rfft(pos_grads.float(), 1)
        pos_fft = torch.view_as_real(torch.fft.rfft(pos_grads.float()))
        pos_mags = torch.norm(pos_fft, dim=2)
        pos_mag_sum = torch.sum(pos_mags, dim=1, keepdim=True)
        pos_mag_sum[pos_mag_sum == 0] = 1  # Keep 0s when the sum is 0
        pos_mags = pos_mags / pos_mag_sum

        # Cut off DC
        pos_mags = pos_mags[:, 1:]

        # Construct weight vector
        weights = place_tensor(torch.ones_like(pos_mags))
        if limit_softness is None:
            weights[:, freq_limit:] = 0
        else:
            x = place_tensor(
                torch.arange(1, pos_mags.size(1) - freq_limit + 1)
            ).float()
            weights[:, freq_limit:] = 1 / (1 + torch.pow(x, limit_softness))

        # Multiply frequency magnitudes by weights
        pos_weighted_mags = pos_mags * weights

        # Add up along frequency axis to get score
        pos_score = torch.sum(pos_weighted_mags, dim=1)
        pos_loss = 1 - pos_score
        return torch.mean(pos_loss)
    else:
        return place_tensor(torch.zeros(1))
