import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    """
    Perform linear predictive analysis of input speech.
    """

    nframes = 1 + (len(speech) - frame_length) // frame_skip

    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))

    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]

        # LPC coefficients
        a = librosa.lpc(frame, order=order)
        A[i, :] = a

        # Prediction error (excitation)
        pred = np.convolve(frame, a)[:frame_length]
        err = pred

        # Only last frame_skip samples are required
        excitation[i, -frame_skip:] = err[-frame_skip:]

    return A, excitation


def synthesize(e, A, frame_skip):
    """
    Synthesize speech from LPC residual and coefficients.
    """

    nframes = A.shape[0]
    order = A.shape[1] - 1

    synthesis = np.zeros(nframes * frame_skip)

    zi = np.zeros(order)

    for i in range(nframes):
        excitation = e[i * frame_skip:(i + 1) * frame_skip]

        y = np.zeros(frame_skip)

        for n in range(frame_skip):
            y[n] = excitation[n]

            for k in range(1, order + 1):
                if n - k >= 0:
                    y[n] -= A[i, k] * y[n - k]
                else:
                    idx = order + (n - k)
                    if idx >= 0:
                        y[n] -= A[i, k] * zi[idx]

        synthesis[i * frame_skip:(i + 1) * frame_skip] = y

        if frame_skip >= order:
            zi = y[-order:]
        else:
            zi = np.concatenate((zi[frame_skip:], y))

    return synthesis


def robot_voice(excitation, T0, frame_skip):
    """
    Create robot-voice excitation.
    """

    gain = np.sqrt(np.mean(excitation ** 2, axis=1))

    nframes = excitation.shape[0]

    e_robot = np.zeros(nframes * frame_skip)

    for i in range(nframes):
        start = i * frame_skip

        for n in range(frame_skip):
            if n % T0 == 0:
                e_robot[start + n] = gain[i]

    return gain, e_robot
