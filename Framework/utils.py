from enum import Enum

	
class Globals:
	'''
		All the global structure required by GOD
	'''
	class Layer(Enum):
		INPUT = 0
		HIDDEN = 1
		OUTPUT = 2

	class Mutation(Enum):
		EDGE = 0
		NODE = 1
		WEIGHT = 2

	def __init__(self, input_features: int, output_features: int, initial_NNs: int):
		self.input_features = input_features
		self.output_features = output_features
		self.map = {}
		self.innovation_number = 0
		self.NNs = []
		self.population = 0
		self.nodes = []
		self.population = initial_NNs