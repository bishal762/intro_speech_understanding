import numpy as np
from scipy.signal import spectrogram

def VAD(waveform, Fs):
    """
    Extract speech segments whose frame energy exceeds 10% of the maximum.
    Frame length = 25 ms
    Frame step = 10 ms
    """

    frame_len = int(0.025 * Fs)
    frame_step = int(0.010 * Fs)

    energies = []
    starts = []

    for start in range(0, len(waveform) - frame_len + 1, frame_step):
        frame = waveform[start:start + frame_len]
        energies.append(np.sum(frame**2))
        starts.append(start)

    energies = np.array(energies)

    if len(energies) == 0:
        return []

    threshold = 0.10 * np.max(energies)
    voiced = energies > threshold

    segments = []

    in_segment = False
    seg_start = 0

    for i, flag in enumerate(voiced):
        if flag and not in_segment:
            seg_start = starts[i]
            in_segment = True

        elif not flag and in_segment:
            seg_end = starts[i] + frame_len
            segments.append(waveform[seg_start:seg_end])
            in_segment = False

    if in_segment:
        segments.append(waveform[seg_start:])

    return segments


def segments_to_models(segments, Fs):
    """
    Pre-emphasize, compute spectrogram,
    keep low-frequency half, average log spectra.
    """

    models = []

    for seg in segments:

        if len(seg) < 2:
            continue

        # Pre-emphasis
        preemph = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])

        frame_len = int(0.004 * Fs)   # 4 ms
        frame_step = int(0.002 * Fs)  # 2 ms
        noverlap = frame_len - frame_step

        f, t, Sxx = spectrogram(
            preemph,
            fs=Fs,
            nperseg=frame_len,
            noverlap=noverlap,
            mode='magnitude'
        )

        # Low-frequency half
        half = Sxx.shape[0] // 2
        Sxx = Sxx[:half, :]

        # Log spectrum
        log_spec = np.log(Sxx + 1e-10)

        # Average over time
        model = np.mean(log_spec, axis=1)

        models.append(model)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    """
    Segment test speech, create test models,
    compare with cosine similarity.
    """

    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    Y = len(models)
    K = len(test_models)

    sims = np.zeros((Y, K))
    test_outputs = []

    for k, test_model in enumerate(test_models):

        best_idx = 0
        best_sim = -np.inf

        for y, model in enumerate(models):

            # Handle possible length mismatch
            L = min(len(model), len(test_model))
            m1 = model[:L]
            m2 = test_model[:L]

            denom = np.linalg.norm(m1) * np.linalg.norm(m2)

            if denom == 0:
                sim = 0
            else:
                sim = np.dot(m1, m2) / denom

            sims[y, k] = sim

            if sim > best_sim:
                best_sim = sim
                best_idx = y

        test_outputs.append(labels[best_idx])

    return sims, test_outputs
