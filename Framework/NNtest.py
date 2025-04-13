from NN import *
from GOD import GOD
from utils import Globals

Universe = Globals(2,2,2)
GOD.create_initial_genomes(Universe)

my_creature = Universe.genomes[0]
GOD.mutate_add_connection(Universe, my_creature)


my_creaturenn = NN(my_creature)

inpu = torch.tensor([1.0, 1.0], requires_grad=True)
print(inpu)
output = my_creaturenn(inpu)
print(output)
x = sum(output)
x.backward()
print(inpu.grad)
