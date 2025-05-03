import streamlit as st
import sys
import pickle
import tempfile
from NN import *
from GOD import *
from utils import Globals
import time

sys.path.append('..')
from Environment.main import *

st.title("Genome Evaluator")

# File uploader
uploaded_file = st.file_uploader("Upload a .pkl Genome File", type=["pkl"])

if uploaded_file is not None:
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Initialize Universe
        st.write("Initializing Universe...")
        Universe2 = Globals(10, 6, POPULATIONS)

        # Load genomes
        st.write("Loading genomes from uploaded file...")
        Universe2.load_genomes(temp_path)

        # Evaluate and mutate
        st.write("Running evaluation and mutation...")
        GOD.Evaluate_and_Mutate(Universe2, False, Visualize=False, Train=False)

        st.success("Process completed successfully.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
