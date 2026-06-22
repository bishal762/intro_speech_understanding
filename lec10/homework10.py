import numpy as np
import torch
import torch.nn as nn
from scipy.signal import spectrogram


def get_features(waveform, Fs):
    '''
    Get spectrogram features and frame labels.
    '''

    # --------------------------------------------------
    # Feature extraction
    # --------------------------------------------------

    # Pre-emphasis
    preemph = np.append(
        waveform[0],
        waveform[1:] - 0.97 * waveform[:-1]
    )

    feat_frame_len = int(0.004 * Fs)   # 4 ms
    feat_frame_step = int(0.002 * Fs)  # 2 ms

    f, t, Sxx = spectrogram(
        preemph,
        fs=Fs,
        nperseg=feat_frame_len,
        noverlap=feat_frame_len - feat_frame_step,
        mode='magnitude'
    )

    # Keep low-frequency half
    Sxx = Sxx[:Sxx.shape[0] // 2, :]

    # Features: one row per frame
    features = Sxx.T

    nframes = features.shape[0]

    # --------------------------------------------------
    # VAD
    # --------------------------------------------------

    vad_frame_len = int(0.025 * Fs)    # 25 ms
    vad_frame_step = int(0.010 * Fs)   # 10 ms

    energies = []
    starts = []

    for start in range(
        0,
        len(waveform) - vad_frame_len + 1,
        vad_frame_step
    ):
        frame = waveform[start:start + vad_frame_len]
        energies.append(np.sum(frame ** 2))
        starts.append(start)

    energies = np.array(energies)

    labels = np.zeros(nframes, dtype=np.int64)

    if len(energies) == 0:
        return features, labels

    threshold = 0.10 * np.max(energies)
    voiced = energies > threshold

    segment_id = 1
    in_segment = False
    seg_start = None

    segments = []

    for i, flag in enumerate(voiced):

        if flag and not in_segment:
            seg_start = starts[i]
            in_segment = True

        elif not flag and in_segment:
            seg_end = starts[i] + vad_frame_len
            segments.append((seg_start, seg_end))
            segment_id += 1
            in_segment = False

    if in_segment:
        segments.append((seg_start, len(waveform)))

    # --------------------------------------------------
    # Assign labels to spectrogram frames
    # --------------------------------------------------

    frame_times = t

    for seg_idx, (start_sample, end_sample) in enumerate(segments):

        start_time = start_sample / Fs
        end_time = end_sample / Fs

        frame_idx = np.where(
            (frame_times >= start_time) &
            (frame_times <= end_time)
        )[0]

        # Repeat each segment label five times
        label = min(seg_idx // 5 + 1, len(segments))

        labels[frame_idx] = label

    return features.astype(np.float32), labels.astype(np.int64)


def train_neuralnet(features, labels, iterations):
    '''
    Train Sequential(LayerNorm, Linear).
    '''

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    nfeats = features.shape[1]
    nlabels = int(torch.max(labels).item()) + 1

    model = nn.Sequential(
        nn.LayerNorm(nfeats),
        nn.Linear(nfeats, nlabels)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    lossvalues = np.zeros(iterations)

    for i in range(iterations):

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        lossvalues[i] = loss.item()

    return model, lossvalues


def test_neuralnet(model, features):
    '''
    Return softmax probabilities.
    '''

    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)

    return probs.detach().numpy()
