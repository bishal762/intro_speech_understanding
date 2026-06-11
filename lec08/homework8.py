import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    '''
    num_frames = int((len(waveform) - frame_length) / step)

    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * step
        frames[i, :] = waveform[start:start + frame_length]

    return frames


def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    '''
    return np.abs(np.fft.fft(frames, axis=1))


def mstft_to_spectrogram(mstft):
    '''
    Convert max(0.001*amax(mstft), mstft) to decibels.
    '''
    floor = 0.001 * np.amax(mstft)
    spectrogram = 20 * np.log10(np.maximum(floor, mstft))
    return spectrogram
