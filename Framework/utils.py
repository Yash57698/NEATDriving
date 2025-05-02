from enum import Enum
from activations import *
import pickle
import os

class Globals:
	'''
		All the global structure required by GOD
	'''
	class Type(Enum):
		INPUT = 0
		HIDDEN = 1
		OUTPUT = 2

	class Mutation(Enum):
		EDGE = 0
		NODE = 1
		WEIGHT = 2

	class Activation(Enum):
		SIGMOID = 0
		RELU = 1
	
	activations = [sigmoid_activation,relu_activation]

	def __init__(self, input_features: int, output_features: int, initial_genomes: int):
		self.current_generation = 1
		self.input_features = input_features
		self.output_features = output_features
		self.connection_map = {}
		self.node_map = {}
		self.innov_num = 0
		self.genomes = []
		self.nodes = input_features + output_features
		self.population = initial_genomes
	
	def save_genomes(self,filepath):
		"""
		Save a list of Genome objects to a file using pickle.
		"""
		with open(filepath, 'wb') as f:
			pickle.dump(self.genomes, f)

	def load_genomes(self,filepath):
		"""
		Load a list of Genome objects from a file using pickle.
		"""
		if not os.path.exists(filepath):
			raise FileNotFoundError(f"No file found at: {filepath}")
		with open(filepath, 'rb') as f:
			self.genomes = pickle.load(f)
		