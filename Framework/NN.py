from utils import Globals

class Node:
	'''
		These are the nodes/neurons in the neural network.
	'''
	def __init__(self, id: int, enabled: bool, bias: float, layer: int):
		self.id = id # uniquely identifies a node
		self.enabled = enabled
		self.bias = bias
		self.layer = layer

class ConnectGenes:
	'''
		These are the connections for the nodes. It is a combination of 2 neurons and a weight.
	'''
	def __init__(self, IN: int, OUT: int, weight: float, enabled: bool, innov_num: int):
		self.IN = IN
		self.OUT = OUT
		self.weight = weight
		self.enabled = enabled
		self.innov = innov_num

class NN:
	'''
		Neural Network.
		These are the individuals in the population.
		GOD will play with them, help them evolve.
		A neural network has a list of nodes.
		It will have connections between the nodes.
	'''
	def __init__(self, globals: Globals):
		'''
			Constructor for creating the basic neural network.
			Basic neural network:
			Input layer : Output layer
			No hidden layers
			No connections
		'''
		self.input_features = globals.input_features
		self.output_features = globals.output_features
		self.nodes = []
		for i in range(1, self.input_features + self.output_features + 1):
			if i <= self.input_features:
				self.nodes.append(Node(i, True, 0, globals.Layer.INPUT))
			else:
				self.nodes.append(Node(i, True, 0, globals.Layer.OUTPUT))
		self.connections = []