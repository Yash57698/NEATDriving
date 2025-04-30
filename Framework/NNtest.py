import sys
from NN import *
from GOD import *
from utils import Globals
sys.path.append('..')
from Environment.main import *

Universe = Globals(8,9,POPULATIONS)
GOD.create_initial_genomes(Universe)
# creature = Universe.genomes[0]
# GOD.mutate_add_connection(Universe, creature)
# GOD.mutate_add_node(Universe, creature)

# GOD.mutate_add_connection(Universe, creature)
# GOD.mutate_add_node(Universe, creature)
# GOD.mutate_add_connection(Universe, creature)

# creature.draw_network_newwindow()
for i in range(10000):
    GOD.Evaluate_and_Mutate(Universe)



# inpu = torch.tensor([1.0, 1.0], requires_grad=True)
# print(inpu)
# output = my_creaturenn(inpu)
# print(output)
# x = sum(output)
# x.backward()
# print(inpu.grad)
