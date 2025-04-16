from enum import Enum
from activations import *
	
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
		self.input_features = input_features
		self.output_features = output_features
		self.connection_map = {}
		self.node_map = {}
		self.innovation_number = 0
		self.genomes = []
		self.population = 0
		self.nodes = []
		self.population = initial_genomes
		