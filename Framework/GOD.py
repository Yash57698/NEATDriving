from utils import Globals
from NN import NN, ConnectGene, Genome

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
		for i in range(len(genome.nodes)):
			for j in range(i+1,len(genome.nodes)):
				if (genome.nodes[i].type == Globals.Type.INPUT and genome.nodes[j].type == Globals.Type.OUTPUT) or \
				   (genome.nodes[i].type == Globals.Type.HIDDEN and genome.nodes[j].type == Globals.Type.HIDDEN) or \
				   (genome.nodes[i].type == Globals.Type.HIDDEN and genome.nodes[j].type == Globals.Type.OUTPUT):
					genome.connections.append(ConnectGene(genome.nodes[i], genome.nodes[j], 1, True, globals.innovation_number))
					globals.innovation_number += 1

	def mutate_add_node(globals: Globals, genome: Genome):
		'''
			GOD said, "Let there be nodes!"
		'''
		pass

	def mutate_weight(globals: Globals, genome: Genome):
		'''
			GOD said, "Let there be change in weight!"
		'''
		pass

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
		pass