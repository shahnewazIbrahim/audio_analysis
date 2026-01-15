import io

import librosa
import numpy as np
import streamlit as st

from audio import analyze_audio, create_visualization_figure

st.set_page_config(page_title="Audio Intelligence Studio", layout="wide")
st.title("Audio Intelligence Studio")
st.write(
    "Upload any spoken or musical audio file and the tool will outline "
    "tempo, pitch, key, loudness scales, and a short briefing. Visualizations "
    "for waveform, spectrogram, and MFCC are shown below."
)

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "flac", "ogg", "m4a"]
)

if uploaded_file is None:
    st.info("Drop an audio file in the upload box to start the analysis.")
else:
    audio_bytes = uploaded_file.read()
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    except Exception as exc:
        st.error(f"Unable to decode the uploaded file: {exc}")
    else:
        st.audio(audio_bytes, format=uploaded_file.type or "audio/wav")

        analysis = analyze_audio(y, sr)
        sound_scale = analysis["sound_scale"]
        key_info = analysis["key_info"]

        cols = st.columns(3)
        cols[0].metric("Tempo (BPM)", f"{analysis['tempo']:.2f}")
        pitch_value = (
            "N/A" if np.isnan(analysis["avg_pitch"]) else f"{analysis['avg_pitch']:.2f} Hz"
        )
        cols[1].metric("Average Pitch", pitch_value)
        cols[2].metric("Detected Key", key_info["scale"])

        st.markdown(f"**Key Confidence:** {key_info['confidence']:.2f}")

        st.subheader("Sound Scale")
        st.write(
            f"- Loudness (RMS): {sound_scale['loudness_scale']}\n"
            f"- Decibel scale: {sound_scale['db_scale']}\n"
            f"- Pitch scale: {sound_scale['pitch_scale']}"
        )

        st.subheader("Audio / Voice Briefing")
        st.code(analysis["brief"])

        st.subheader("Visualizations")
        fig = create_visualization_figure(y, sr)
        st.pyplot(fig)
