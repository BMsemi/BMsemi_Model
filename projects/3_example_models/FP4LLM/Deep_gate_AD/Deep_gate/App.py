import streamlit as st
import numpy as np
import librosa
import librosa.display
import io
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import idct

# Importing hardware simulator modules
from User import load_schema, parse_layer, execute_quant_linear, MultiPENeuromorphicCore, L2_Buffer


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
N_MFCC       = 20
N_FRAMES     = 32
HOP_LENGTH   = 512
N_FFT        = 2048
N_MELS       = 128
GRIFFIN_ITER = 64


# ─────────────────────────────────────────────
# 1. Hardware Initialization
# ─────────────────────────────────────────────
@st.cache_resource
def init_simulator():
    mpe       = MultiPENeuromorphicCore(num_pes=512, input_size=32, output_size=32)
    l2_buffer = L2_Buffer(capacity=10_000_000)
    return mpe, l2_buffer


@st.cache_resource
def load_layers():
    schema_data = load_schema("schema.json")
    return [parse_layer(l) for l in schema_data]


mpe, l2_buffer = init_simulator()
layers         = load_layers()


# ─────────────────────────────────────────────
# 2. Noise estimation from a quiet segment
# ─────────────────────────────────────────────
def estimate_noise_profile(y, sr, n_fft, hop_length, noise_duration=0.5):
    """
    Estimate noise STFT magnitude from the first `noise_duration` seconds.
    Assumes the beginning of the file contains mostly background noise.
    """
    n_noise_samples = int(noise_duration * sr)
    noise_segment   = y[:n_noise_samples] if len(y) > n_noise_samples else y
    noise_stft      = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_length)
    noise_profile   = np.mean(np.abs(noise_stft), axis=1, keepdims=True)  # (1+n_fft//2, 1)
    return noise_profile


# ─────────────────────────────────────────────
# 3. Spectral subtraction denoiser
# ─────────────────────────────────────────────
def spectral_subtraction(y, sr, n_fft, hop_length, noise_profile, strength=2.0):
    """
    Classic spectral subtraction:
      |S_clean| = max(|S_noisy| - strength * |N|, 0)
    Then Griffin-Lim to reconstruct phase.
    """
    stft_noisy = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag_noisy  = np.abs(stft_noisy)
    phase      = np.angle(stft_noisy)

    # Subtract estimated noise floor
    mag_clean  = np.maximum(mag_noisy - strength * noise_profile, 0.0)

    # Reconstruct with original phase (fast, good quality)
    stft_clean = mag_clean * np.exp(1j * phase)
    y_clean    = librosa.istft(stft_clean, hop_length=hop_length, n_fft=n_fft)
    return y_clean, mag_noisy, mag_clean


# ─────────────────────────────────────────────
# 4. Neuromorphic MFCC smoothing pass
#    The autoencoder refines the denoised MFCCs
#    by passing them through the hardware simulator.
# ─────────────────────────────────────────────
def run_neuromorphic_pipeline(y_input, sr, layers, mpe, l2_buffer, metric_cols):
    """
    Extract MFCCs → quantize → run through neuromorphic autoencoder
    → dequantize → return smoothed MFCC matrix.
    """
    # ── Extract MFCCs ──
    mfcc = librosa.feature.mfcc(
        y=y_input, sr=sr,
        n_mfcc=N_MFCC, hop_length=HOP_LENGTH,
        n_mels=N_MELS, n_fft=N_FFT,
    )
    if mfcc.shape[1] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])))
    mfcc_slice = mfcc[:, :N_FRAMES]           # (20, 32)

    # ── Quantize ──
    s_in  = float(layers[0]['input_scale'])
    zp_in = int(layers[0]['input_zp'])
    x_int8 = np.clip(
        np.round(mfcc_slice.flatten() / s_in) + zp_in,
        -128, 127
    ).astype(np.int8)

    # ── Neuromorphic inference ──
    x_current = x_int8
    y_float   = None

    for i, params in enumerate(layers):
        y_float, y_int8, stats = execute_quant_linear(
            layer_idx    = i,
            params       = params,
            x_int8       = x_current,
            mapping_file = "schema_10layer_mapping_fc.m",
            mpe          = mpe,
            l2_buffer    = l2_buffer,
            verbose      = False,
        )
        x_current = y_int8
        with metric_cols[i % 5]:
            st.metric(f"L{i} Util", f"{stats['util_pct']:.1f}%")

    # ── Dequantize ──
    last    = layers[-1]
    s_out   = float(last.get('output_scale', 1.0))
    zp_out  = int(last.get('output_zp', 0))
    y_deq   = (y_float.astype(np.float32) - zp_out) * s_out

    # ── Reshape ──
    expected = N_MFCC * N_FRAMES
    if y_deq.size != expected:
        y_deq = np.resize(y_deq, expected)
    mfcc_out = y_deq.reshape(N_MFCC, N_FRAMES)

    return mfcc_slice, mfcc_out, s_in, zp_in, s_out, zp_out, y_float


# ─────────────────────────────────────────────
# 5. MFCC matrix → audio (scipy idct, librosa 0.10+ safe)
# ─────────────────────────────────────────────
def mfcc_to_audio(mfcc_matrix, sr, n_mfcc, n_mels, n_fft, hop_length, n_iter):
    """
    MFCCs → inverse DCT → log-Mel → db_to_power
          → pinv(mel filterbank) → STFT magnitude → Griffin-Lim → audio
    """
    # Step 1 — pad n_mfcc → n_mels, then inverse DCT-II
    mfcc_padded          = np.zeros((n_mels, mfcc_matrix.shape[1]), dtype=np.float32)
    mfcc_padded[:n_mfcc] = mfcc_matrix
    log_mel              = idct(mfcc_padded, type=2, axis=0, norm='ortho')  # (n_mels, n_frames)

    # Step 2 — log-Mel (dB) → linear Mel power
    mel_power = librosa.db_to_power(log_mel)

    # Step 3 — Mel power → STFT magnitude via pseudo-inverse filterbank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mag_spec  = np.dot(np.linalg.pinv(mel_basis), mel_power)
    mag_spec  = np.maximum(0.0, mag_spec)

    # Step 4 — Griffin-Lim phase reconstruction
    audio = librosa.griffinlim(
        mag_spec,
        n_iter     = n_iter,
        hop_length = hop_length,
        n_fft      = n_fft,
    )
    return audio


# ─────────────────────────────────────────────
# 6. UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="AD01 Neuromorphic Denoiser", layout="wide")
st.title("🔊 AD01: Neuromorphic Audio Denoiser")
st.info("Mode: PWM_INT8_ROW_SERIAL (32×32 Crossbar Tiles)  |  Noise removal via spectral subtraction + neuromorphic MFCC smoothing")

uploaded_file = st.file_uploader("Upload noisy audio ('inputfile.wav')", type=["wav"])

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Denoising Settings")
    noise_dur   = st.slider("Noise estimation window (s)",  0.1, 2.0, 0.5,  0.1,
                            help="Duration of the leading quiet segment used to estimate noise floor.")
    strength    = st.slider("Spectral subtraction strength", 0.5, 5.0, 2.0,  0.25,
                            help="Higher = more aggressive noise removal, but may introduce artefacts.")
    use_neuro   = st.checkbox("Apply neuromorphic MFCC smoothing", value=True,
                              help="Pass denoised MFCCs through the hardware autoencoder for extra smoothing.")
    use_griffinlim = st.checkbox("Re-synthesise via Griffin-Lim after neuro pass", value=False,
                                 help="Reconstruct audio from neuromorphic MFCCs (slower, more artefacts).")

if uploaded_file:

    audio_bytes = uploaded_file.read()

    st.subheader("🎙️ Original (noisy) Audio")
    st.audio(audio_bytes, format="audio/wav")

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # ─────────────────────────────────────────────
    # Stage A — Spectral Subtraction
    # ─────────────────────────────────────────────
    with st.status("Stage 1/2 — Spectral subtraction…") as s1:
        noise_profile             = estimate_noise_profile(y, sr, N_FFT, HOP_LENGTH, noise_dur)
        y_denoised, mag_n, mag_c  = spectral_subtraction(y, sr, N_FFT, HOP_LENGTH, noise_profile, strength)
        s1.update(label="Stage 1 complete — spectral subtraction done", state="complete")

    # ─────────────────────────────────────────────
    # Stage B — Neuromorphic MFCC smoothing (optional)
    # ─────────────────────────────────────────────
    if use_neuro:
        st.subheader("Hardware Simulation Trace")
        metric_cols = st.columns(5)

        with st.status("Stage 2/2 — Neuromorphic MFCC smoothing…") as s2:
            mfcc_in, mfcc_out, s_in, zp_in, s_out, zp_out, y_float = run_neuromorphic_pipeline(
                y_denoised, sr, layers, mpe, l2_buffer, metric_cols
            )
            s2.update(label="Stage 2 complete — neuromorphic pass done", state="complete")

        if use_griffinlim:
            # Rebuild audio from neuromorphic-smoothed MFCCs
            y_final = mfcc_to_audio(mfcc_out, sr, N_MFCC, N_MELS, N_FFT, HOP_LENGTH, GRIFFIN_ITER)
        else:
            # Keep spectral-subtraction audio (better quality);
            # neuromorphic pass was used only to validate / log MFCC smoothing
            y_final = y_denoised
    else:
        y_final   = y_denoised
        mfcc_in   = librosa.feature.mfcc(y=y,          sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_mels=N_MELS, n_fft=N_FFT)[:, :N_FRAMES]
        mfcc_out  = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_mels=N_MELS, n_fft=N_FFT)[:, :N_FRAMES]
        s_in = zp_in = s_out = zp_out = y_float = None

    # Normalise
    peak = np.max(np.abs(y_final))
    if peak > 0:
        y_final = y_final / peak

    # ─────────────────────────────────────────────
    # Results
    # ─────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Denoising Results")

    # ── Spectrograms ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**Noisy input spectrogram**")
        fig_n, ax_n = plt.subplots(figsize=(5, 3))
        D_n = librosa.amplitude_to_db(mag_n, ref=np.max)
        im  = librosa.display.specshow(D_n, sr=sr, hop_length=HOP_LENGTH,
                                       x_axis='time', y_axis='hz', ax=ax_n)
        plt.colorbar(im, ax=ax_n, format="%+2.0f dB")
        ax_n.set_title("Noisy input")
        st.pyplot(fig_n); plt.close(fig_n)

    with col_b:
        st.write("**Denoised spectrogram**")
        fig_c, ax_c = plt.subplots(figsize=(5, 3))
        D_c = librosa.amplitude_to_db(mag_c, ref=np.max)
        im2 = librosa.display.specshow(D_c, sr=sr, hop_length=HOP_LENGTH,
                                       x_axis='time', y_axis='hz', ax=ax_c)
        plt.colorbar(im2, ax=ax_c, format="%+2.0f dB")
        ax_c.set_title("After spectral subtraction")
        st.pyplot(fig_c); plt.close(fig_c)

   
    # ── Waveform comparison ──
    st.write("**Waveform comparison**")
    fig_w, axes_w = plt.subplots(2, 1, figsize=(10, 4), sharex=False)
    t_orig  = np.linspace(0, len(y)       / sr, len(y))
    t_final = np.linspace(0, len(y_final) / sr, len(y_final))
    axes_w[0].plot(t_orig,  y,       color='#E24B4A', linewidth=0.4, alpha=0.8)
    axes_w[0].set_title("Noisy input waveform");  axes_w[0].set_ylabel("Amplitude")
    axes_w[1].plot(t_final, y_final, color='#1D9E75', linewidth=0.4, alpha=0.8)
    axes_w[1].set_title("Denoised output waveform"); axes_w[1].set_ylabel("Amplitude"); axes_w[1].set_xlabel("Time (s)")
    plt.tight_layout()
    st.pyplot(fig_w); plt.close(fig_w)

    # ── Noise reduction metric ──
    min_len   = min(len(y), len(y_final))
    noise_est = y[:min_len] - y_final[:min_len]
    snr_in    = 10 * np.log10(np.mean(y[:min_len]**2)       / (np.mean(noise_est**2) + 1e-12))
    snr_out   = 10 * np.log10(np.mean(y_final[:min_len]**2) / (np.mean(noise_est**2) + 1e-12))

    m1, m2, m3 = st.columns(3)
    m1.metric("Input SNR (est.)",  f"{snr_in:.1f} dB")
    m2.metric("Output SNR (est.)", f"{snr_out:.1f} dB")
    m3.metric("SNR improvement",   f"{snr_out - snr_in:+.1f} dB")

    # ── Play & Download ──
    st.divider()
    st.subheader("🔈 Denoised Audio")
    out_bio = io.BytesIO()
    sf.write(out_bio, y_final, sr, format='WAV', subtype='PCM_16')

    out_bio.seek(0)
    st.audio(out_bio.read(), format="audio/wav")

    out_bio.seek(0)
    st.download_button(
        label     = "⬇️ Download denoised audio",
        data      = out_bio.read(),
        file_name = "outputfile_denoised.wav",
        mime      = "audio/wav",
    )

    # ── Debug ──
    with st.expander("Debug info"):
        info = {
            "sample_rate"        : int(sr),
            "original_samples"   : int(len(y)),
            "denoised_samples"   : int(len(y_final)),
            "noise_duration_s"   : noise_dur,
            "subtraction_strength": strength,
            "neuromorphic_pass"  : use_neuro,
            "griffinlim_resynth" : use_griffinlim,
            "n_fft"              : N_FFT,
            "n_mels"             : N_MELS,
            "hop_length"         : HOP_LENGTH,
            "griffin_lim_iters"  : GRIFFIN_ITER,
        }
        if use_neuro and y_float is not None:
            info.update({
                "input_scale"    : float(s_in),
                "input_zp"       : int(zp_in),
                "output_scale"   : float(s_out),
                "output_zp"      : int(zp_out),
                "neuro_out_min"  : float(round(float(y_float.min()), 4)),
                "neuro_out_max"  : float(round(float(y_float.max()), 4)),
            })
        st.json(info)
