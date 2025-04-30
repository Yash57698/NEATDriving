from utils import Globals
from NN import NN, ConnectGene, Genome, Node
import random
import torch
import numpy as np
from copy import deepcopy
from config import *
import sys
sys.path.append('..')
from Environment.main import RunRound
from Environment.main_parallel import CarSimulator
from config import *

specieinfotex = ""
sim = CarSimulator()
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
		p = np.random.choice([0, 1], p=[0.9, 0.1])
		if p == 0:
			conn.weight += random.normalvariate(0, 0.75)
		else:
			conn.weight = torch.tensor(random.normalvariate(0, 1))

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

	def genetic_distance(genome1: Genome, genome2: Genome):
		'''
			GOD said, "Let there be distance!"
		'''
		distance = 0
		genome1_conn = sorted(genome1.connections, key = lambda c: c.innov)
		genome2_conn = sorted(genome2.connections, key = lambda c: c.innov)
		N = max(len(genome1_conn), len(genome2_conn)) + 1
		E = 0
		D = 0
		w1 = 0
		w2 = 0
		i = 0
		j = 0
		while i < len(genome1_conn) and j < len(genome2_conn):
			if genome1_conn[i].innov == genome2_conn[j].innov:
				w1 += genome1_conn[i].weight.item()
				w2 += genome2_conn[j].weight.item()
				i += 1
				j += 1
			elif genome1_conn[i].innov < genome2_conn[j].innov:
				w1 += genome1_conn[i].weight.item()
				D += 1
				i += 1
			else:
				w2 += genome2_conn[j].weight.item()
				D += 1
				j += 1
		while i < len(genome1_conn):
			w1 += genome1_conn[i].weight.item()
			E += 1
			i += 1
		while j < len(genome2_conn):
			w2 += genome2_conn[j].weight.item()
			E += 1
			j += 1
		
		w1 /= len(genome1_conn) if len(genome1_conn) > 0 else 1
		w2 /= len(genome2_conn) if len(genome2_conn) > 0 else 1

		W = abs(w1 - w2) / max(w1, w2) if max(w1, w2) != 0 else 0

		return (C1 * E + C2 * D) / N + C3 * W

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
		global specieinfotex
		Representative_genome = [globals.genomes[0]]
		for i in globals.genomes:
			flag = False
			for j in Representative_genome:
				if GOD.genetic_distance(i, j) < SPECIES_THRESHOLD:
					flag = True
			if( not flag):
				Representative_genome.append(i)

		species = [[] for i in range(len(Representative_genome))]
		for i in globals.genomes:
			flag = False
			for j in Representative_genome:
				if GOD.genetic_distance(i, j) < SPECIES_THRESHOLD:
					species[Representative_genome.index(j)].append(i)
					flag = True
					break
			if not flag:
				Representative_genome.append(i)
				species.append([i])

		species_fitness = [0 for i in range(len(species))]
		for ind,speca in enumerate(species):
			for indx,j in enumerate(speca):
				score = sim.run(NN(j, grad = False), globals.current_generation, genome = j,id = indx,speciesid = ind,species_infotext=specieinfotex , framecap=150 + 10 * (globals.current_generation//10))
				# score = RunRound(NN(j, grad = False), globals.current_generation, genome = j,id = indx,speciesid = ind,species_infotext=specieinfotex , framecap=150 + 10 * (globals.current_generation//10))
				speca[indx] = [score/len(speca), speca[indx]]
				species_fitness[ind] += score/len(speca)
			species[ind] = [species_fitness[ind], species[ind]]

		newGeneration = []
		species.sort(reverse = True, key = lambda x: x[0])
		cutoff = len(species)
		curr = 0
		for i in range(len(species)):
			if curr > globals.population * 0.8:
				cutoff = i
				break
			curr += len(species[i])
		# print(species)
		species = species[:cutoff]
		species_fitness = [i[0] for i in species]
		total_fitness = sum(species_fitness)
		specieinfotex = "No of Species :" + str(len(species)) + "\n"
		for i in range(len(species)):
			specieinfotex += "Species " + str(i) + " : " + str(len(species[i][1])) + f"fitness : {species[i][0]}" + "\n"

		for ind,specas in enumerate(species):
			speca = specas[1]
			speca.sort(reverse = True, key = lambda x: x[0])
			new_species_population = int((species_fitness[ind]/total_fitness) * globals.population)
			if new_species_population != 0:
				for i in range(new_species_population):
					no_to_keep = len(speca)//2
					if no_to_keep == 0:
						no_to_keep = 1
					ch = np.random.choice([0,1,2], p=[0.25,0.55,0.20])
					gen = random.choice(speca[:no_to_keep])[1]
					if len(gen.connections) <= 2:
						ch = 1
					if ch == 0:
						genome = deepcopy(gen)
					elif ch == 1:
						genome = deepcopy(gen)
						if len(genome.connections) <=2:
							GOD.mutate(globals, genome, Globals.Mutation.EDGE)
						else:
							GOD.mutate(globals, genome, np.random.choice([Globals.Mutation.EDGE, Globals.Mutation.NODE, Globals.Mutation.WEIGHT],p=[MUTATE_CONNECTION, MUTATE_NODE, MUTATE_WEIGHT]))
					else:
						parent1 = gen
						parent2 = random.choice(speca[:no_to_keep])[1]
						genome = GOD.let_there_be_sex(globals, parent1, parent2)
					newGeneration.append(genome)
		globals.current_generation += 1
		globals.genomes = newGeneration