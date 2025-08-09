import torchaudio
import torchaudio.transforms as T
import torch
import matplotlib.pyplot as plt
import librosa
from torch.nn.utils.rnn import pad_sequence


def load_fbank(audio_path, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path)

    assert sr == sample_rate, f"Sample rate mismatch! {audio_path} ({sr}) should be {sample_rate}"

    transform = T.MelSpectrogram(
        sample_rate=sr, 
        n_fft=400, 
        win_length=400,  # 25ms at 16kHz
        hop_length=160,  # 10ms at 16kHz
        n_mels=80,
        window_fn=torch.hamming_window
    )
    
    fbank = transform(waveform)
    
    # Convert to log scale
    fbank = torch.log(fbank + 1e-9)
    
    fbank = fbank.squeeze(0)  # (80, time)
    
    # Stack 4 frames with skip rate of 3
    max_frames = fbank.shape[1] - (4-1) * 3
    if max_frames <= 0:
        raise ValueError("Audio too short for frame stacking")
    
    # Create indices for stacked frames
    indices = torch.arange(max_frames).unsqueeze(1) + torch.arange(0, 4) * 3
    stacked = fbank[:, indices]  # (80, time, 4)
    stacked = stacked.permute(1, 0, 2).reshape(max_frames, -1)  # (time, 320)

    stacked = (stacked - stacked.mean(dim=0, keepdim=True)) / (stacked.std(dim=0, keepdim=True) + 1e-9)
    
    return stacked  # (T, 320)


def build_label_map(df):
    unique_labels = df[['action', 'object', 'location']].astype(str).agg('-'.join, axis=1).unique()
    label2id = {l: i for i, l in enumerate(sorted(unique_labels))}
    return label2id


def pad_collate(batch):
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True)  # (B, T, 320)
    ys = torch.tensor(ys)
    return xs, ys


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


if __name__ == "__main__":
    test_file = "fluent_speech_commands_dataset/wavs/speakers/5o9BvRGEGvhaeBwA/0cc59730-44eb-11e9-a1ea-79ca03012c0e.wav"
    signal, sr = torchaudio.load(test_file)
    fig, axs = plt.subplots(2, 1)
    plot_waveform(signal, sr, title="Original waveform", ax=axs[0])

    spect = load_fbank(test_file)
    plot_spectrogram(spect, title="Mel Spectrogram", ax=axs[1])

    fig.tight_layout()
    plt.show()
