import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pretty_midi
import soundfile as sf
import os

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
instrument_name = "Acoustic Grand Piano"  # Set instrument for MIDI output

# Function to convert MIDI to WAV
def midi_to_wav(midi_file: str, wav_file: str):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    audio_data = midi_data.fluidsynth()  # Use FluidSynth to synthesize audio
    sf.write(wav_file, audio_data, 16000)  # Save as WAV at 16kHz

# Function to create MIDI from notes
def notes_to_midi(notes_df: pd.DataFrame, out_file: str, instrument_name: str):
    midi = pretty_midi.PrettyMIDI()
    # Create an instrument instance
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))
    
    for index, row in notes_df.iterrows():
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

# Streamlit app UI
st.title("AI-Powered Music Note Generator")
st.write("Generate musical note sequences using a neural network model trained on MIDI data.")

# User input parameters
temperature = st.slider("Temperature (randomness level)", min_value=0.1, max_value=3.0, value=1.0)
num_predictions = st.slider("Number of Notes to Generate", min_value=10, max_value=200, value=120)
instrument_choice = st.selectbox("Choose Instrument", ["Acoustic Grand Piano", "Violin", "Electric Guitar", "Flute"])

# Create an initial sample input sequence to use for note generation
sample_notes = np.random.randint(60, 72, (seq_length, 3)) / np.array([vocab_size, 1, 1])  # Mock data

# Generate notes button
if st.button("Generate Notes"):
    generated_notes = []
    prev_start = 0
    input_notes = sample_notes.copy()

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

    # Save to DataFrame
    generated_notes_df = pd.DataFrame(generated_notes, columns=['pitch', 'step', 'duration', 'start', 'end'])
    st.write("Generated Notes Data:", generated_notes_df.head(10))

    # Generate MIDI
    midi_file = 'generated_output.mid'
    midi_output = notes_to_midi(generated_notes_df, out_file=midi_file, instrument_name=instrument_choice)

    # Convert MIDI to WAV
    wav_file = 'generated_output.wav'
    midi_to_wav(midi_file, wav_file)

    # Check if the WAV file exists and its size
    if os.path.exists(wav_file):
        if os.path.getsize(wav_file) > 0:
            st.audio(wav_file)  # Play the generated WAV audio
        else:
            st.error("WAV file is empty!")
    else:
        st.error("WAV file not found!")

    # Optional: You can plot the piano roll or distributions here using similar functions
    # st.pyplot(plot_piano_roll(generated_notes_df))
    # st.pyplot(plot_distributions(generated_notes_df))
