from utils import Globals
from NN import NN, ConnectGene, Genome, Node
import random

class GOD:
	'''
		GOD.
		He will be the one who manages every neural network there is.
		He will be the one who mutates the neural networks.
		He will be the one who lets them have sex.
	'''	
	def create_initial_genomes(globals: Globals):
		'''
			GOD said, "Let there be NNs!".
		'''
		for i in range(globals.population):
			# NNs should create their nodes for now
			if i < globals.input_features:
				globals.genomes.append(Genome(globals))
			else:
				globals.genomes.append(Genome(globals))

	def mutate_add_connection(globals: Globals, genome: Genome):
		'''
			GOD said, "Let there be edges!"
		'''
		weight = random.normalvariate(0, 1)
		done = False
		while not done:
			node1, node2 = random.sample(genome.nodes, 2)
			if ((node1.type == Globals.Type.OUTPUT and node2.type == Globals.Type.OUTPUT) or (node1.type == Globals.Type.INPUT and node2.type == Globals.Type.INPUT) or (node1.type == Globals.Type.OUTPUT) or (node2.type == Globals.Type.INPUT)):
				continue
			if (node1.id, node2.id) in globals.connection_map: # If these nodes have a innov number already
				if (node1.id, node2.id) in [(c.IN.id, c.OUT.id) for c in genome.connections]: # If these nodes have a connection already
					continue
				genome.connections.append(ConnectGene(node1, node2, weight, True, globals.map[(node1.id, node2.id)]))
			else:
				globals.innov_num += 1 # Create a new innov number
				genome.connections.append(ConnectGene(node1, node2, weight, True, globals.innov_num))
				globals.connection_map[(node1.id, node2.id)] = globals.innov_num
			done = True

	def mutate_add_node(globals: Globals, genome: Genome):
		'''
			GOD said, "Let there be nodes!"
		'''
		if len(genome.connections) == 0:
			return
		
		done = False
		while not done:
			conn = random.choice(genome.connections)
			if not conn.enabled:
				continue
			if (conn.IN.id, conn.OUT.id) in globals.node_map:
				new_node = globals.node_map[conn.IN.id, conn.OUT.id]
				print("adding a node between ", conn.IN.id, "and", conn.OUT.id)
				genome.nodes.append(new_node)
				conn1 = ConnectGene(conn.IN, new_node, 1, True, globals.innov_num)
				conn2 = ConnectGene(new_node, conn.OUT, conn.weight, True, globals.innov_num)
				genome.connections.append(conn1)
				genome.connections.append(conn2)
				conn.enabled = False
				done = True
			else:
				node = Node(globals.nodes, True, random.normalvariate(0, 1), Globals.Type.HIDDEN)
				globals.nodes += 1
				genome.nodes.append(node)
				print("adding a node between ", conn.IN.id, "and", conn.OUT.id)
				globals.innov_num += 1
				conn1 = ConnectGene(conn.IN, node, 1, True, globals.innov_num)
				globals.innov_num += 1
				conn2 = ConnectGene(node, conn.OUT, conn.weight, True, globals.innov_num)
				genome.connections.append(conn1)
				genome.connections.append(conn2)
				globals.node_map[(conn.IN.id, conn.OUT.id)] = node
				conn.enabled = False
				done = True

	def mutate_weight(globals: Globals, genome: Genome):
		'''
			GOD said, "Let there be change in weight!"
		'''
		if len(genome.connections) == 0:
			return
		conn = random.choice(genome.connections)
		conn.weight += random.normalvariate(0, 0.1)

	def mutate(globals: Globals, genome: Genome, kind: int):
		'''
			GOD said, "Let there be mutation!"
		'''
		if kind == Globals.Mutation.EDGE:
			GOD.mutate_add_connection(globals, genome)
		elif kind == Globals.Mutation.NODE:
			GOD.mutate_add_node(globals, genome)
		elif kind == Globals.Mutation.WEIGHT:
			GOD.mutate_weight(globals, genome)

	def let_there_be_sex(self, globals: Globals, genome1: Genome, genome2: Genome):
		'''
			GOD said, "Let there be sex!"
		'''
		sorted(genome1.connections, key = lambda c: c.innov_num)
		sorted(genome2.connections, key = lambda c: c.innov_num)
		genome = Genome(globals)
		genome.nodes = list(set(genome1.nodes).union(set(genome2.nodes)))
		i = 0
		j = 0
		while i < len(genome1.connections) and j < len(genome2.connections):
			if genome1.connections[i].innov_num == genome2.connections[j].innov_num:
				genome.connections.append(
					ConnectGene(genome1.connections[i].IN, genome1.connections[i].OUT, genome1.connections[i].weight, genome1.connections[i].enabled and genome2.connections[i].enabled, genome1.connections[i].innov_num)
				)
				i += 1
				j += 1
			elif genome1.connections[i].innov_num < genome2.connections[j].innov_num:
				genome.connections.append(
					ConnectGene(genome1.connections[i].IN, genome1.connections[i].OUT, genome1.connections[i].weight, genome1.connections[i].enabled, genome1.connections[i].innov_num)
				)
				i += 1
			elif genome1.connections[i].innov_num > genome2.connections[j].innov_num:
				genome.connections.append(
					ConnectGene(genome2.connections[i].IN, genome2.connections[i].OUT, genome2.connections[i].weight, genome2.connections[i].enabled, genome2.connections[i].innov_num)
				)
				j += 1
		globals.genomes.append(genome)
			
		
