import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.
    '''
    n = np.arange(Fs // 2)  # one-half second

    f3 = f * (2 ** (4/12))  # major third
    f5 = f * (2 ** (7/12))  # perfect fifth

    x = (
        np.cos(2 * np.pi * f * n / Fs) +
        np.cos(2 * np.pi * f3 * n / Fs) +
        np.cos(2 * np.pi * f5 * n / Fs)
    )

    return x


def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    '''
    k = np.arange(N).reshape(N, 1)
    n = np.arange(N).reshape(1, N)

    W = np.cos(2 * np.pi * k * n / N) - 1j * np.sin(2 * np.pi * k * n / N)

    return W.astype(complex)


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    X = np.fft.fft(x)
    mag = np.abs(X)

    # Use only positive-frequency half of FFT
    mag[len(mag)//2:] = 0

    freqs = []

    for _ in range(3):
        idx = np.argmax(mag)
        freqs.append(idx * Fs / len(x))
        mag[idx] = 0

    freqs.sort()

    return freqs[0], freqs[1], freqs[2]
