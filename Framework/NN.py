from utils import Globals
import torch
import networkx as nx
import matplotlib.pyplot as plt

class Node:
	'''
		These are the nodes/neurons in the neural network.
	'''
	def __init__(self, id: int, enabled: bool, bias: float, type: int):
		self.id = id # uniquely identifies a node
		self.enabled = enabled
		self.bias = bias
		self.type = type
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

	# def draw_network(self):
	# 	neurons = [node.id for node in self.nodes]
	# 	input_neurons = [node.id for node in self.nodes if node.type == Globals.Type.INPUT]
	# 	output_neurons = [node.id for node in self.nodes if node.type == Globals.Type.OUTPUT]
	# 	connections = [(conn.IN.id, conn.OUT.id) for conn in self.connections if conn.enabled]
	# 	G = nx.DiGraph()
	# 	G.add_nodes_from(neurons)
	# 	G.add_edges_from(connections)

	# 	# Initialize spring layout for base layout
	# 	pos = nx.kamada_kawai_layout(G)

	# 	hidden_neurons = [n for n in neurons if n not in input_neurons and n not in output_neurons]

	# 	# Spread inputs vertically on the left (x = -1)
	# 	for i, n in enumerate(sorted(input_neurons)):
	# 		pos[n] = (-1, 1 - 2 * i / max(len(input_neurons) - 1, 1))

	# 	# Spread outputs vertically on the right (x = 1)
	# 	for i, n in enumerate(sorted(output_neurons)):
	# 		pos[n] = (1, 1 - 2 * i / max(len(output_neurons) - 1, 1))

	# 	plt.figure(figsize=(10, 7))
	# 	nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue')
	# 	nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
	# 	nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
	# 	plt.title("Neural Network Layout (Inputs Left, Outputs Right)")
	# 	plt.axis('off')

	# 	plt.savefig("neural_networks.png")
	# 	plt.show()
	def draw_network(self,name = "Network"):
		import collections

		neurons = [node.id for node in self.nodes]
		input_neurons = [node.id for node in self.nodes if node.type == Globals.Type.INPUT]
		output_neurons = [node.id for node in self.nodes if node.type == Globals.Type.OUTPUT]
		hidden_neurons = [node.id for node in self.nodes if node.id not in input_neurons + output_neurons]

		connections = [(conn.IN.id, conn.OUT.id) for conn in self.connections if conn.enabled]

		G = nx.DiGraph()
		G.add_nodes_from(neurons)
		G.add_edges_from(connections)

		# --- Step 1: Compute topological depth ---
		def compute_depths():
			depths = {nid: 0 for nid in neurons}
			visited = set()

			def dfs(nid, depth):
				if nid in visited:
					depths[nid] = max(depths[nid], depth)
				else:
					visited.add(nid)
					depths[nid] = depth
				for _, tgt in G.edges(nid):
					dfs(tgt, depth + 1)

			for nid in input_neurons:
				dfs(nid, 0)

			return depths

		depths = compute_depths()

		# --- Step 2: Set X coordinates ---
		pos = {}

		# Inputs: always at x = -1
		for nid in input_neurons:
			pos[nid] = (-1.0, 0)

		# Outputs: always at x = 1
		for nid in output_neurons:
			pos[nid] = (1.0, 0)

		# Hidden: strictly between -1 and 1
		if hidden_neurons:
			min_d = min(depths[nid] for nid in hidden_neurons)
			max_d = max(depths[nid] for nid in hidden_neurons)
			range_d = max(max_d - min_d, 1)

			for nid in hidden_neurons:
				# Normalize and squeeze into (-1, 1)
				normalized = (depths[nid] - min_d) / range_d
				x = -0.9 + 1.8 * normalized  # ensures x in (-1, 1)
				pos[nid] = (x, 0)

		# --- Step 3: Assign Y coordinates by layering nodes vertically at each X ---
		x_buckets = collections.defaultdict(list)
		for nid, (x, _) in pos.items():
			x_buckets[x].append(nid)

		for x, nids in x_buckets.items():
			n = len(nids)
			for i, nid in enumerate(sorted(nids)):
				y = 1 - 2 * i / max(n - 1, 1)
				pos[nid] = (x, y)

		# --- Step 4: Draw ---
		plt.figure(figsize=(12, 8))
		nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue')
		nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20)
		nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

		plt.title("Neural Network Layout (Inputs Left, Outputs Right, Hidden Strictly Between)")
		plt.axis('off')
		plt.tight_layout()
		plt.savefig(f"{name}.png")
		plt.show()

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



