import sys
from NN import *
from GOD import *
from utils import Globals
import time
sys.path.append('..')
from Environment.main import *

Universe2 = Globals(10,6,POPULATIONS)
Universe2.load_genomes("./Populations/genomes_generationstill30.pkl")
GOD.Evaluate_and_Mutate(Universe2,False,Visualize=True,Train = False)