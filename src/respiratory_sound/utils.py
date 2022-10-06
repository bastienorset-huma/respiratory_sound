import math
import numpy as np

# Continous Wavelet Transform with Morlet wavelet 
# Original code by Alexander Neergaard, https://github.com/neergaard/CWT
# 
# Parameters:
#   data: input data
#   nv: # of voices (scales) per octave
#   sr: sampling frequency (Hz)
#   low_freq: lowest frequency (Hz) of interest (limts longest scale)
def cwt2(data, nv=10, sr=1., low_freq=0.):
    data -= np.mean(data)
    n_orig = data.size
    ds = 1 / nv
    dt = 1 / sr

    # Pad data symmetrically
    padvalue = n_orig // 2
    x = np.concatenate((np.flipud(data[0:padvalue]), data, np.flipud(data[-padvalue:])))
    n = x.size

    # Define scales
    _, _, wavscales = getDefaultScales(n_orig, ds, sr, low_freq)
    num_scales = wavscales.size

    # Frequency vector sampling the Fourier transform of the wavelet
    omega = np.arange(1, math.floor(n / 2) + 1, dtype=np.float64)
    omega *= (2 * np.pi) / n
    omega = np.concatenate((np.array([0]), omega, -omega[np.arange(math.floor((n - 1) / 2), 0, -1, dtype=int) - 1]))

    # Compute FFT of the (padded) time series
    f = np.fft.fft(x)

    # Loop through all the scales and compute wavelet Fourier transform
    psift, freq = waveft(omega, wavscales)

    # Inverse transform to obtain the wavelet coefficients.
    cwtcfs = np.fft.ifft(np.kron(np.ones([num_scales, 1]), f) * psift)
    cfs = cwtcfs[:, padvalue:padvalue + n_orig]
    freq = freq * sr

    return cfs, freq

def getDefaultScales(n, ds, sr, low_freq):
    nv = 1 / ds
    # Smallest useful scale (default 2 for Morlet)
    s0 = 2

    # Determine longest useful scale for wavelet
    max_scale = n // (np.sqrt(2) * s0)
    if max_scale <= 1:
        max_scale = n // 2
    max_scale = np.floor(nv * np.log2(max_scale)) 
    a0 = 2 ** ds
    scales = s0 * a0 ** np.arange(0, max_scale + 1)
    
    # filter out scales below low_freq
    fourier_factor = 6 / (2 * np.pi)
    frequencies = sr * fourier_factor / scales
    frequencies = frequencies[frequencies >= low_freq]
    scales = scales[0:len(frequencies)]

    return s0, ds, scales

def waveft(omega, scales):
    num_freq = omega.size
    num_scales = scales.size
    wft = np.zeros([num_scales, num_freq])

    gC = 6
    mul = 2
    for jj, scale in enumerate(scales):
        expnt = -(scale * omega - gC) ** 2 / 2 * (omega > 0)
        wft[jj, ] = mul * np.exp(expnt) * (omega > 0)

    fourier_factor = gC / (2 * np.pi)
    frequencies = fourier_factor / scales

    return wft, frequencies


def get_max_length_from_list(list_seq,axis=1):
    max_len = 0
    for seq in list_seq:
        len = np.shape(seq)[axis]
        if len > max_len:
            max_len = len
    return max_len

def convert_list_arrays_to_3d_array(list_seq):
    max_len = get_max_length_from_list(list_seq,axis=1)
    n_sample = len(list_seq)
    n_freq = np.shape(list_seq[0])[0]
    X_data = np.zeros((n_sample,n_freq,max_len))
    for i, seqs in enumerate(list_seq):
        len_seq = np.shape(seqs)[1]
        X_data[i,:,:len_seq] = seqs 
    return X_data