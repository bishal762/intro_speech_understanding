import numpy as np
from scipy import signal


def voiced_excitation(duration, F0, Fs):
    '''
    Create voiced speeech excitation.
    
    @param:
    duration (scalar) - length of the excitation, in samples
    F0 (scalar) - pitch frequency, in Hertz
    Fs (scalar) - sampling frequency, in samples/second
    
    @returns:
    excitation (np.ndarray) - the excitation signal
    '''

    excitation = np.zeros(duration)

    # Number of samples between pitch pulses
    period = int(np.round(Fs / F0))

    # Put -1 at every integer multiple of the period
    excitation[::period] = -1

    return excitation


def resonator(x, F, BW, Fs):
    '''
    Generate the output of a resonator.
    
    @param:
    x (np.ndarray(N)) - the excitation signal
    F (scalar) - resonant frequency, in Hertz
    BW (scalar) - resonant bandwidth, in Hertz
    Fs (scalar) - sampling frequency, in samples/second
    
    @returns:
    y (np.ndarray(N)) - resonant output
    '''

    # Pole radius from bandwidth
    r = np.exp(-np.pi * BW / Fs)

    # Resonant frequency angle
    theta = 2 * np.pi * F / Fs

    # Second-order resonator denominator
    a = np.array([
        1,
        -2 * r * np.cos(theta),
        r ** 2
    ])

    # Numerator
    b = np.array([1])

    # Filter excitation through resonator
    y = signal.lfilter(b, a, x)

    return y


def synthesize_vowel(duration,F0,F1,F2,F3,F4,BW1,BW2,BW3,BW4,Fs):
    '''
    Synthesize a vowel.
    
    @returns:
    speech (np.ndarray(samples)) - synthesized vowel
    '''

    # Create voiced excitation
    excitation = voiced_excitation(duration, F0, Fs)

    # Apply four formant resonators
    speech = resonator(excitation, F1, BW1, Fs)
    speech = resonator(speech, F2, BW2, Fs)
    speech = resonator(speech, F3, BW3, Fs)
    speech = resonator(speech, F4, BW4, Fs)

    return speech
