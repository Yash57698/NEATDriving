from utils import Globals
from NN import NN, ConnectGene, Genome, Node
import random
import numpy as np
from copy import deepcopy
from config import *
import sys
sys.path.append('..')
from Environment.main import RunRound

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
				genome.connections.append(ConnectGene(node1, node2, weight, True, globals.connection_map[(node1.id, node2.id)]))
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
				# print("adding a node between ", conn.IN.id, "and", conn.OUT.id)
				print(new_node.id, "already exists")
				new_node.parent1id = conn.IN.id
				new_node.parent2id = conn.OUT.id
				genome.nodes.append(new_node)
				conn1 = ConnectGene(conn.IN, new_node, 1, True, globals.innov_num)
				conn2 = ConnectGene(new_node, conn.OUT, conn.weight, True, globals.innov_num)
				genome.connections.append(conn1)
				genome.connections.append(conn2)
				conn.enabled = False
				done = True
			else:
				new_node = Node(globals.nodes, random.normalvariate(0, 1), Globals.Type.HIDDEN)
				globals.nodes += 1
				genome.nodes.append(new_node)
				new_node.parent1id = conn.IN.id
				new_node.parent2id = conn.OUT.id
				# print("adding a node between ", conn.IN.id, "and", conn.OUT.id)
				print("new node id: ", new_node.id)
				globals.innov_num += 1
				conn1 = ConnectGene(conn.IN, new_node, 1, True, globals.innov_num)
				globals.innov_num += 1
				conn2 = ConnectGene(new_node, conn.OUT, conn.weight, True, globals.innov_num)
				genome.connections.append(conn1)
				genome.connections.append(conn2)
				globals.node_map[(conn.IN.id, conn.OUT.id)] = new_node
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

	def let_there_be_sex(globals: Globals, genome1: Genome, genome2: Genome):
		'''
			GOD said, "Let there be sex!"
		'''
		sorted(genome1.connections, key = lambda c: c.innov)
		sorted(genome2.connections, key = lambda c: c.innov)
		genome = Genome(globals)
		for node in genome1.nodes:
			if node.type != Globals.Type.INPUT and node.type != Globals.Type.OUTPUT:
				genome.nodes.append(Node(node.id, node.bias, node.type))
		for node in genome2.nodes:
			if node.type != Globals.Type.INPUT and node.type != Globals.Type.OUTPUT:
				genome.nodes.append(Node(node.id, node.bias, node.type))
		i = 0
		j = 0
		while i < len(genome1.connections) and j < len(genome2.connections):
			# print("i: ", i, "j: ", j, "len1: ", len(genome1.connections), "len2: ", len(genome2.connections), "innov1: ", genome1.connections[i].innov, "innov2: ", genome2.connections[j].innov)
			if genome1.connections[i].innov == genome2.connections[j].innov:
				genome.connections.append(
					ConnectGene(genome1.connections[i].IN, genome1.connections[i].OUT, genome1.connections[i].weight, genome1.connections[i].enabled and genome2.connections[j].enabled, genome1.connections[i].innov)
				)
				i += 1
				j += 1
			elif genome1.connections[i].innov < genome2.connections[j].innov:
				genome.connections.append(
					ConnectGene(genome1.connections[i].IN, genome1.connections[i].OUT, genome1.connections[i].weight, genome1.connections[i].enabled, genome1.connections[i].innov)
				)
				i += 1
			elif genome1.connections[i].innov > genome2.connections[j].innov:
				genome.connections.append(
					ConnectGene(genome2.connections[j].IN, genome2.connections[j].OUT, genome2.connections[j].weight, genome2.connections[j].enabled, genome2.connections[j].innov)
				)
				j += 1
		while i < len(genome1.connections):
			genome.connections.append(
				ConnectGene(genome1.connections[i].IN, genome1.connections[i].OUT, genome1.connections[i].weight, genome1.connections[i].enabled, genome1.connections[i].innov)
			)
			i += 1
		while j < len(genome2.connections):
			
			genome.connections.append(
				ConnectGene(genome2.connections[j].IN, genome2.connections[j].OUT, genome2.connections[j].weight, genome2.connections[j].enabled, genome2.connections[j].innov)
			)
			j += 1

		

		return genome
			
	def Evaluate_and_Mutate(globals: Globals):
		'''
			GOD said, "Let there be evaluation!"
		'''
		parentGeneration = []
		for (indx,genome) in enumerate(globals.genomes):
			parentGeneration.append((RunRound(NN(genome, grad = False),globals.current_generation, id = indx,genome=genome),genome))	

		globals.current_generation += 1		

		parentGeneration.sort(reverse = True, key = lambda x: x[0])

		# parentGeneration[0][1].draw_network("best")

		newGeneration = []
		for i in range(int(globals.population * (1-POPULATION_TO_DESTROY))):
			newGeneration.append(deepcopy(parentGeneration[i][1]))

		to_be_mutated = random.sample(parentGeneration[:int(globals.population)], int(globals.population * POPULATION_TO_MUTATE))

		for genome in to_be_mutated:
			GOD.mutate(globals, genome[1], np.random.choice([Globals.Mutation.EDGE, Globals.Mutation.NODE, Globals.Mutation.WEIGHT],p=[MUTATE_CONNECTION, MUTATE_NODE, MUTATE_WEIGHT]))
		
		for genome in to_be_mutated:
			newGeneration.append(deepcopy(genome[1]))

		to_be_mated = random.sample(newGeneration, int(globals.population * POPULATION_TO_MATE)*2)
		for i in range(0, len(to_be_mated), 2):
			newGeneration.append(GOD.let_there_be_sex(globals, to_be_mated[i], to_be_mated[i+1]))

		globals.genomes = newGeneration[:globals.population]