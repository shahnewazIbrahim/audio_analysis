import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def get_sound_scale(y, sr, avg_pitch_hz):
    # RMS loudness
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_db = float(librosa.amplitude_to_db(np.array([rms_mean]), ref=1.0)[0])

    # Loudness scale
    if rms_mean < 0.02:
        loudness = "Very Quiet"
    elif rms_mean < 0.05:
        loudness = "Quiet"
    elif rms_mean < 0.1:
        loudness = "Moderate"
    elif rms_mean < 0.2:
        loudness = "Loud"
    else:
        loudness = "Very Loud"

    # dB scale
    if rms_db < -40:
        db_scale = "Very Soft"
    elif rms_db < -25:
        db_scale = "Soft"
    elif rms_db < -15:
        db_scale = "Normal"
    elif rms_db < -5:
        db_scale = "Loud"
    else:
        db_scale = "Very Loud"

    # Frequency / pitch scale
    if np.isnan(avg_pitch_hz) or avg_pitch_hz <= 0:
        pitch_scale = "Pitch not detected"
    elif avg_pitch_hz < 150:
        pitch_scale = "Low-pitch voice"
    elif avg_pitch_hz < 300:
        pitch_scale = "Mid-pitch voice"
    else:
        pitch_scale = "High-pitch voice"

    return {
        "rms_value": rms_mean,
        "rms_db": rms_db,
        "loudness_scale": loudness,
        "db_scale": db_scale,
        "pitch_scale": pitch_scale
    }

def detect_musical_key(y, sr):
    # Compute chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Note names (librosa order)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Find dominant note
    root_index = int(np.argmax(chroma_mean))
    root_note = notes[root_index]

    # Major & minor key profiles (Krumhansl-Schmuckler)
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
         2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )

    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
         2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    # Rotate profiles to match root
    def rotate(profile, n):
        return np.roll(profile, n)

    major_score = np.corrcoef(chroma_mean, rotate(major_profile, root_index))[0, 1]
    minor_score = np.corrcoef(chroma_mean, rotate(minor_profile, root_index))[0, 1]

    mode = "Major" if major_score >= minor_score else "Minor"
    confidence = max(major_score, minor_score)

    return {
        "root_note": root_note,
        "scale": f"{root_note} {mode}",
        "confidence": confidence
    }


def generate_voice_brief(y, sr, tempo, avg_pitch_hz, f0, voiced_flag):
    # Duration
    duration = librosa.get_duration(y=y, sr=sr)

    # RMS loudness (overall energy)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_db = float(librosa.amplitude_to_db(np.array([rms_mean]), ref=1.0)[0])

    # Voice activity ratio (how much of the time voice is present)
    # voiced_flag can be bool array; handle None safely
    if voiced_flag is not None and len(voiced_flag) > 0:
        voice_ratio = float(np.mean(voiced_flag))  # 0..1
    else:
        voice_ratio = 0.0

    # Simple noise estimate using spectral flatness (higher -> more noise-like)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    flatness_mean = float(np.mean(flatness))

    # Simple SNR-ish estimate:
    # Use median RMS as "noise floor" proxy and max/mean as signal proxy
    rms_median = float(np.median(rms))
    snr_like_db = 20 * np.log10((rms_mean + 1e-9) / (rms_median + 1e-9))

    # Dominant frequency band from average spectrum
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    avg_spec = np.mean(S, axis=1)
    dom_idx = int(np.argmax(avg_spec))
    dom_freq = float(freqs[dom_idx])

    # Pitch-based guess (very rough)
    pitch_note = None
    if not np.isnan(avg_pitch_hz) and avg_pitch_hz > 0:
        pitch_note = librosa.hz_to_note(avg_pitch_hz)

    

    # Classification-ish text
    # (These thresholds are heuristics — not a medical/forensic claim)
    clarity = "clear" if flatness_mean < 0.25 else "somewhat noisy" if flatness_mean < 0.4 else "noisy"
    consistency = "consistent" if voice_ratio > 0.55 else "intermittent" if voice_ratio > 0.25 else "sparse"
    energy_desc = "moderate" if -35 < rms_db < -15 else "low" if rms_db <= -35 else "high"

    # Tempo note
    tempo_note = ""
    if tempo is not None and not np.isnan(tempo):
        if tempo < 60:
            tempo_note = "slow rhythmic activity (or speech/non-musical content)"
        elif tempo <= 120:
            tempo_note = "moderate rhythmic activity"
        else:
            tempo_note = "fast rhythmic activity"
    else:
        tempo_note = "tempo not available"

    # Build briefing text (English)
    lines = []
    lines.append("===== AUDIO / VOICE BRIEFING =====")
    lines.append(f"Duration: {duration:.2f} seconds")
    lines.append(f"Sampling Rate: {sr} Hz")
    lines.append(f"Estimated Tempo: {tempo:.2f} BPM ({tempo_note})" if tempo is not None and not np.isnan(tempo) else "Estimated Tempo: N/A")

    if not np.isnan(avg_pitch_hz):
        if pitch_note:
            lines.append(f"Average Pitch (voiced): {avg_pitch_hz:.2f} Hz (~{pitch_note})")
        else:
            lines.append(f"Average Pitch (voiced): {avg_pitch_hz:.2f} Hz")
    else:
        lines.append("Average Pitch (voiced): No clear pitch detected")

    lines.append(f"Voice Activity (voiced frames): {voice_ratio*100:.1f}% ({consistency} speaking presence)")
    lines.append(f"Average Loudness (RMS): {rms_mean:.4f} (~{rms_db:.1f} dBFS approx)")
    lines.append(f"Dominant Frequency (avg spectrum peak): ~{dom_freq:.1f} Hz")
    lines.append(f"Spectral Flatness (noise-likeness): {flatness_mean:.3f} → audio is {clarity}")
    lines.append(f"SNR-like Estimate: {snr_like_db:.1f} dB (higher usually means cleaner separation)")

    # A short natural language summary
    summary = (
        f"Overall, the audio appears {clarity} with {energy_desc} loudness and a "
        f"{consistency} voice presence. The spectral energy is concentrated around typical voice ranges, "
        f"and pitch tracking {'worked well' if not np.isnan(avg_pitch_hz) else 'did not find a stable pitch'}."
    )
    lines.append("")
    lines.append("Summary:")
    lines.append(summary)

    return "\n".join(lines)


# ------------------ MAIN ------------------
filename = "your_audio_file.wav"

try:
    y, sr = librosa.load(filename)
    print(f"Audio loaded successfully!\nDuration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
    print(f"Sampling rate: {sr} Hz")
except FileNotFoundError:
    print("Error: File not found. Please provide the correct file name or path.")
    raise SystemExit

# Tempo (handle array/scalar)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
if isinstance(tempo, np.ndarray):
    tempo = tempo.item() if tempo.size == 1 else float(np.mean(tempo))
tempo = float(tempo) if tempo is not None else np.nan
print(f"Estimated tempo (BPM): {tempo:.2f}")

# Pitch (F0)
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7")
)
avg_pitch = float(np.nanmean(f0)) if f0 is not None else np.nan
print(f"Average pitch (Frequency): {avg_pitch:.2f} Hz" if not np.isnan(avg_pitch) else "No clear pitch detected\n")

sound_scale = get_sound_scale(y, sr, avg_pitch)

print("===== SOUND SCALE =====")
print(f"Loudness (RMS): {sound_scale['loudness_scale']}")
print(f"Volume Scale (dB): {sound_scale['db_scale']}")
print(f"Pitch Scale: {sound_scale['pitch_scale']}\n")


key_info = detect_musical_key(y, sr)

print("===== MUSICAL SCALE / KEY =====")
print(f"Detected Scale: {key_info['scale']}")
print(f"Confidence Score: {key_info['confidence']:.2f}")

# ---- Generate briefing ----
brief = generate_voice_brief(y, sr, tempo, avg_pitch, f0, voiced_flag)
print("\n" + brief + "\n")

# ---- Visualization ----
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.title("Waveform (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel-frequency Spectrogram")

plt.subplot(3, 1, 3)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
librosa.display.specshow(mfccs, x_axis="time")
plt.colorbar()
plt.title("MFCC")

plt.tight_layout()
plt.show()
