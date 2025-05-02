import sys
from NN import *
from GOD import *
from utils import Globals
import time
sys.path.append('..')
from Environment.main import *

Universe = Globals(10,6,POPULATIONS)
GOD.create_initial_genomes(Universe)
# creature = Universe.genomes[0]
# GOD.mutate_add_connection(Universe, creature)
# GOD.mutate_add_node(Universe, creature)

# GOD.mutate_add_connection(Universe, creature)
# GOD.mutate_add_node(Universe, creature)
# GOD.mutate_add_connection(Universe, creature)

# creature.draw_network_newwindow()
# for i in range(10000):
#     GOD.Evaluate_and_Mutate(Universe)
for i in range(10000):
    start_time = time.time()
    GOD.Evaluate_and_Mutate(Universe,True,Visualize=False)
    if i%10 == 0:
        Universe.save_genomes(f"genomes_generationstill{i}.pkl")
    print(f"Generation {i+1} completed in time: {time.time() - start_time} seconds")
end_time = time.time()
print(f"Time taken to run Evaluate_and_Mutate: {end_time - start_time} seconds")

# Universe.save_genomes("genomes.pkl")
# Universe2 = Globals(10,6,POPULATIONS)
# Universe2.load_genomes("genomes.pkl")
# GOD.Evaluate_and_Mutate(Universe2,False)


# start_time = time.time()
# GOD.Evaluate_and_Mutate(Universe,True)
# end_time = time.time()
# print(f"Time taken to run Evaluate_and_Mutate: {end_time - start_time} seconds")


# inpu = torch.tensor([1.0, 1.0], requires_grad=True)
# print(inpu)
# output = my_creaturenn(inpu)
# print(output)
# x = sum(output)
# x.backward()
# print(inpu.grad)
