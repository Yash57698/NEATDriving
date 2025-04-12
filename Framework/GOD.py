from utils import Globals
from NN import NN, ConnectGenes

class GOD:
	'''
		GOD.
		He will be the one who manages every neural network there is.
		He will be the one who mutates the neural networks.
		He will be the one who lets them have sex.
	'''	
	def create_initial_NNs(self, globals: Globals):
		'''
			GOD said, "Let there be NNs!".
		'''
		for i in range(globals.population):
			# NNs should create their nodes for now
			if i < globals.input_features:
				globals.NNs.append(NN(globals))
			else:
				globals.NNs.append(NN(globals))

	def mutate_add_connection(self, globals: Globals, NN: NN):
		'''
			GOD said, "Let there be edges!"
		'''
		for i in range(len(NN.nodes)):
			for j in range(len(NN.nodes)):
				if i != j:
					NN.connections.append(ConnectGenes(NN.nodes[i].id, NN.nodes[j].id, 0, True, globals.innovation_number))
					globals.innovation_number += 1

	def mutate_add_node(self, globals: Globals, NN: NN):
		'''
			GOD said, "Let there be nodes!"
		'''
		pass

	def mutate_weight(self, globals: Globals, NN: NN):
		'''
			GOD said, "Let there be change in weight!"
		'''
		pass

	def mutate(self, globals: Globals, NN: NN, kind: int):
		'''
			GOD said, "Let there be mutation!"
		'''
		if kind == Globals.Mutation.EDGE:
			self.mutate_add_connection(globals, NN)
		elif kind == Globals.Mutation.NODE:
			self.mutate_add_node(globals, NN)
		elif kind == Globals.Mutation.WEIGHT:
			self.mutate_weight(globals, NN)

	def have_sex(self, globals: Globals, NN1: NN, NN2: NN):
		'''
			GOD said, "Let there be sex!"
		'''
		pass