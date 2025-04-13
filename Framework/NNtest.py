from NN import *
from GOD import GOD
from utils import Globals

Universe = Globals(2,2,2)
GOD.create_initial_genomes(Universe)

my_creature = Universe.genomes[0]
GOD.mutate_add_connection(my_creature)


my_creaturenn = NN(my_creature)

inpu = torch.rand(2, requires_grad=True)
output = my_creaturenn(inpu)
output.backward()
print(inpu.grad)

