import os
import glob
import argparse
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true', default=False, help='perform inference on GPU')
args = parser.parse_args()
gpu = args.gpu

if gpu == False:
  file = open("cpu_timing.txt", "w")
  file.write("CPU only inference timing \n")

  file.write("\n 2 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=2GeV --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("2 GeV complete")

  file = open("cpu_timing.txt", "a")
  file.write("\n 1.5 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=1GeV5 --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("1.5 GeV complete")

  file = open("cpu_timing.txt", "a")
  file.write("\n 1 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=1GeV --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("1 GeV complete")
  """
  file = open("cpu_timing.txt", "a")
  file.write("\n 0.75 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=0GeV75 --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("0.75 GeV complete")

  file = open("cpu_timing.txt", "a")
  file.write("\n 0.6 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=0GeV6 --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("0.6 GeV complete")

  file = open("cpu_timing.txt", "a")
  file.write("\n 0.5 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=0GeV5 --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("0.5 GeV complete")
  """
else:
  file = open("gpu_timing.txt", "w")
  file.write("CPU-GPU inference timing \n")

  file.write("\n 2 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=2GeV --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("2 GeV complete")

  file = open("gpu_timing.txt", "a")
  file.write("\n 1.5 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=1GeV5 --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("1.5 GeV complete")

  file = open("gpu_timing.txt", "a")
  file.write("\n 1 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=1GeV --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("1 GeV complete")
  """
  file = open("gpu_timing.txt", "a")
  file.write("\n 0.75 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=0GeV75 --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("0.75 GeV complete")

  file = open("gpu_timing.txt", "a")
  file.write("\n 0.6 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=0GeV6 --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("0.6 GeV complete")

  file = open("gpu_timing.txt", "a")
  file.write("\n 0.5 GeV - \n")
  file.close()
  os.system("python timeit_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=0GeV5 --construction=heptrkx_plus --lr=0.005 --gamma=0.9")
  print("0.5 GeV complete")
  """
