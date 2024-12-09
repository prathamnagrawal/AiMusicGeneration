import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pretty_midi
import soundfile as sf
import os
from io import BytesIO

# Define custom loss function
@tf.keras.utils.register_keras_serializable()
def mse_with_positive_pressure(y_true, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    pressure = tf.reduce_mean(tf.maximum(0.0, -y_pred))  # example pressure term
    return mse + pressure

# Load the trained model (update the path as necessary)
model = tf.keras.models.load_model('my_model.keras', custom_objects={'mse_with_positive_pressure': mse_with_positive_pressure})

# Define parameters
vocab_size = 128  # Adjust based on your vocabulary
seq_length = 25   # Length of note sequence for input
instrument_name = "Acoustic Grand Piano"  # Default instrument for MIDI output

# Function to convert MIDI to WAV
def midi_to_wav(midi_data: pretty_midi.PrettyMIDI, wav_file: str):
    audio_data = midi_data.fluidsynth()  # Use FluidSynth to synthesize audio
    sf.write(wav_file, audio_data, 16000)  # Save as WAV at 16kHz

# Function to create MIDI from notes
def notes_to_midi(notes_df: pd.DataFrame, out_file: str, instrument_name: str):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))
    
    for _, row in notes_df.iterrows():
        note = pretty_midi.Note(
            velocity=100,  # Set a default velocity for notes
            pitch=int(row['pitch']),
            start=row['start'],
            end=row['end']
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(out_file)  # Save MIDI file
    return out_file

# Function for predicting the next note
def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0) -> tuple[int, float, float]:
    assert temperature > 0
    inputs = tf.expand_dims(notes, 0)
    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # Ensure non-negative values for step and duration
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)

# Extract notes from a MIDI file as a DataFrame
def midi_to_notes(midi_data) -> pd.DataFrame:
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append([note.pitch, note.start, note.end - note.start])  # duration is end - start
    df = pd.DataFrame(notes, columns=['pitch', 'start', 'duration'])
    df['end'] = df['start'] + df['duration']  # Calculate 'end' column
    return df

# Streamlit app UI
st.title("AI-Powered Music Note Generator")
st.write("Generate the next sequence of notes from an uploaded MIDI file using a neural network model.")

# User input parameters
temperature = st.slider("Temperature (randomness level)", min_value=0.1, max_value=3.0, value=1.0)
num_predictions = st.slider("Number of Additional Notes to Generate", min_value=10, max_value=200, value=120)
instrument_choice = st.selectbox("Choose Instrument", ["Acoustic Grand Piano", "Violin", "Electric Guitar", "Flute"])
generation_mode = st.radio("Choose Generation Mode", ("Extend Original MIDI", "Generate New Sequence Only"))

# File uploader for the initial MIDI
uploaded_file = st.file_uploader("Upload a MIDI file", type=["mid", "midi"])

if uploaded_file is not None:
    # Load the uploaded MIDI file
    midi_data = pretty_midi.PrettyMIDI(BytesIO(uploaded_file.read()))
    st.write("Uploaded MIDI file loaded successfully.")

    # Play the uploaded MIDI file
    uploaded_wav_file = "uploaded_midi.wav"
    midi_to_wav(midi_data, uploaded_wav_file)
    if os.path.exists(uploaded_wav_file):
        st.write("Playing the uploaded MIDI file:")
        st.audio(uploaded_wav_file)

    # Convert MIDI to notes DataFrame
    original_notes_df = midi_to_notes(midi_data)
    st.write("Original Notes Data:", original_notes_df.head(10))

    # Initialize sample notes based on the uploaded MIDI data
    sample_notes = original_notes_df[['pitch', 'start', 'duration']].values
    input_notes = np.array(sample_notes[:seq_length])  # Use the first seq_length notes

    generated_notes = []
    prev_start = input_notes[-1, 1] if len(input_notes) > 0 else 0  # Last note's start time

    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        
        # Update the input notes for the next prediction
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    # Save generated notes to DataFrame
    generated_notes_df = pd.DataFrame(generated_notes, columns=['pitch', 'step', 'duration', 'start', 'end'])
    st.write("Generated Notes Data:", generated_notes_df.head(10))

    # Handle output based on generation mode
    if generation_mode == "Extend Original MIDI":
        # Concatenate original and generated notes
        combined_notes_df = pd.concat([
            original_notes_df[['pitch', 'start', 'duration', 'end']],
            generated_notes_df[['pitch', 'start', 'duration', 'end']]
        ]).reset_index(drop=True)
    else:
        # Use only the generated notes
        combined_notes_df = generated_notes_df

    # Generate MIDI from combined notes
    midi_file = 'generated_output.mid'
    midi_output = notes_to_midi(combined_notes_df, out_file=midi_file, instrument_name=instrument_choice)

    # Convert MIDI to WAV
    wav_file = 'generated_output.wav'
    midi_to_wav(pretty_midi.PrettyMIDI(midi_file), wav_file)

    # Check if the WAV file exists and its size
    if os.path.exists(wav_file):
        if os.path.getsize(wav_file) > 0:
            st.write("Playing the generated output:")
            st.audio(wav_file)  # Play the generated WAV audio
        else:
            st.error("WAV file is empty!")
    else:
        st.error("WAV file not found!")
