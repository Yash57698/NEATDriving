from utils import Globals
import torch
import networkx as nx
import matplotlib.pyplot as plt
import pygame

class Node:
	'''
		These are the nodes/neurons in the neural network.
	'''
	def __init__(self, id: int, enabled: bool, bias: float, type: int):
		self.id = id # uniquely identifies a node
		self.enabled = enabled
		self.bias = bias
		self.type = type
		self.parent1id = -1
		self.parent2id = -1
		self.activation = 0
		self.activationtype = 0
		self.computed = False

class ConnectGene:
	'''
		These are the connections for the nodes. It is a combination of 2 neurons and a weight.
	'''
	def __init__(self, IN: Node, OUT: Node, weight: float, enabled: bool, innov_num: int):
		self.IN = IN
		self.OUT = OUT
		self.weight = weight
		self.enabled = enabled
		self.innov = innov_num

class Genome:
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
		for i in range(self.input_features + self.output_features):
			if i < self.input_features:
				self.nodes.append(Node(i, True, 0, globals.Type.INPUT))
			else:
				self.nodes.append(Node(i, True, 0, globals.Type.OUTPUT))
		self.connections = []

	def draw_network(self,name = "Network",screen = None, offset = (0,0)):
		neurons = [node.id for node in self.nodes]
		input_neurons = [node.id for node in self.nodes if node.type == Globals.Type.INPUT]
		output_neurons = [node.id for node in self.nodes if node.type == Globals.Type.OUTPUT]
		hidden_neurons = [(node.id,node.parent1id,node.parent2id) for node in self.nodes if node.id not in input_neurons + output_neurons]
		connections = [(conn.IN.id, conn.OUT.id) for conn in self.connections if conn.enabled]

		# Create a surface instead of initializing a screen
		surface_width = 300
		surface_height = 500

		# Colors
		white = (255, 255, 255)
		black = (0, 0, 0)
		blue = (0, 0, 255)
		red = (255, 0, 0)
		green = (0, 255, 0)

		# Node positions
		positions = {}

		# Calculate positions for input neurons (stacked on the left)
		for i, n in enumerate(sorted(input_neurons)):
			x = 0
			y = 0 + i * (surface_height - 200) // max(len(input_neurons) - 1, 1)
			positions[n] = (x, y)

		# Calculate positions for output neurons (stacked on the right)
		for i, n in enumerate(sorted(output_neurons)):
			x = surface_width
			y = 0 + i * (surface_height - 200) // max(len(output_neurons) - 1, 1)
			positions[n] = (x, y)

		# Calculate positions for hidden neurons (average of their parents)
		def calculate_position(n):
			if type(n) == int:
				return positions[n]
			if n[0] in positions:
				return positions[n[0]]
			parents = []
			if n[1] != -1:
				parents.append(n[1])
			if n[2] != -1:
				parents.append(n[2])
			if parents:
				x = sum(calculate_position(p)[0] for p in parents) / len(parents)
				y = sum(calculate_position(p)[1] for p in parents) / len(parents)
				positions[n[0]] = (x, y)
			else:
				positions[n[0]] = (surface_width // 2, surface_width // 2)  # Default position if no parents
			return positions[n[0]]

		for n in hidden_neurons:
			calculate_position(n)
		

		# Draw connections
		for conn in self.connections:
			if conn.enabled:
				start_pos = positions[conn.IN.id]
				end_pos = positions[conn.OUT.id]
				color = green if conn.weight > 0 else red
				# print(offset + start_pos, offset + end_pos)
				pygame.draw.line(screen, color, (offset[0] + start_pos[0] ,offset[1] + start_pos[1]), (offset[0] + end_pos[0] ,offset[1] + end_pos[1]), 2)

		# Draw neurons
		for n, pos in positions.items():
			color = blue if n in input_neurons else (red if n in output_neurons else black)
			pygame.draw.circle(screen, color, (offset[0] + int(pos[0]), offset[1] + int(pos[1])), 5)
		return

	def draw_network_newwindow(self,name = "Network"):

		neurons = [node.id for node in self.nodes]
		input_neurons = [node.id for node in self.nodes if node.type == Globals.Type.INPUT]
		output_neurons = [node.id for node in self.nodes if node.type == Globals.Type.OUTPUT]
		hidden_neurons = [(node.id,node.parent1id,node.parent2id) for node in self.nodes if node.id not in input_neurons + output_neurons]
		# print(input_neurons,output_neurons,hidden_neurons)
		connections = [(conn.IN.id, conn.OUT.id) for conn in self.connections if conn.enabled]

		
		pygame.init()

		# Screen dimensions
		screen_width = 800
		screen_height = 600
		screen = pygame.display.set_mode((screen_width, screen_height))
		pygame.display.set_caption(name)

		# Colors
		white = (255, 255, 255)
		black = (0, 0, 0)
		blue = (0, 0, 255)
		red = (255, 0, 0)
		green = (0, 255, 0)

		# Node positions
		positions = {}
		running = True
		while running:
			screen.fill(white)

			# Draw connections
			# for conn in self.connections:
			# 	if conn.enabled:
			# 		start_pos = positions[conn.IN.id]
			# 		end_pos = positions[conn.OUT.id]
			# 		color = green if conn.weight > 0 else red
			# 		pygame.draw.line(screen, color, start_pos, end_pos, 2)

			# # Draw neurons
			# for n, pos in positions.items():
			# 	color = blue if n in input_neurons else (red if n in output_neurons else black)
			# 	pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), 20)
			self.draw_network(screen=screen,offset=(0,0))
			pygame.display.flip()

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False

		pygame.quit()

class NN(torch.nn.Module):
	'''
		Neural Network.
		This is the neural network that will be used to train the population.
		It will be used to evaluate the fitness of the population.
		It will be used to play the game.
		It will be used to evolve the population.
	'''
	def __init__(self, genome: Genome, grad = True):
		super(NN, self).__init__()

		self.genome = genome
		connections = genome.connections

		for conn in connections:
			conn.weight = torch.nn.Parameter(torch.tensor([conn.weight], dtype=torch.float64), requires_grad=grad)

		for node in genome.nodes:
			node.bias = torch.nn.Parameter(torch.tensor([node.bias], dtype=torch.float64), requires_grad=grad)

		revadjacencylist = {}
		for node in genome.nodes:
			revadjacencylist[node.id] = []

		for connection in connections:
			revadjacencylist[connection.OUT.id].append(connection)
		self.revadjacencylist = revadjacencylist

		self.connections = connections
		

	def forward(self, input: torch.Tensor):
		for node in self.genome.nodes:
			node.activation = 0
			node.computed = False

		for i in range(self.genome.input_features):
			self.genome.nodes[i].activation = input[i]
			self.genome.nodes[i].computed = True

		
		def get_activation(node):
			if node.computed:
				return node.activation
			
			node.activation = torch.Tensor([0.0])
			# print(self.revadjacencylist[node.id])
			for conn in self.revadjacencylist[node.id]:
				if conn.enabled:
					node.activation += get_activation(conn.IN) * conn.weight
					# print(conn.IN.activation,conn.weight)
			node.activation += node.bias
			# node.activation = Globals.activations[node.activationtype](node.activation)

			node.computed = True
			return node.activation

		for node in self.genome.nodes:
			if(node.type == Globals.Type.OUTPUT):
				# print(node.id)
				get_activation(node)

		return [self.genome.nodes[i].activation for i in range(len(self.genome.nodes)) if self.genome.nodes[i].type == Globals.Type.OUTPUT]



