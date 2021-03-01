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
  file.write("CPU only inference timing \n \n")

  file.write("2 GeV - ")
  file.close()
  os.system("python run_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=2GeV --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9 --save-model")
  print("2 GeV complete")
"""
  file = open("cpu_timing.txt", "a")
  file.write("1.5 GeV - ")
  file.close()
  os.system("python run_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=1GeV5 --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9 --save-model")
  print("1.5 GeV complete")

  file = open("cpu_timing.txt", "a")
  file.write("1 GeV - ")
  file.close()
  os.system("python run_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=1GeV --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9 --save-model")
  print("1 GeV complete")

  file = open("cpu_timing.txt", "a")
  file.write("0.75 GeV - ")
  file.close()
  os.system("python run_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=0GeV75 --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9 --save-model")
  print("0.75 GeV complete")

  file = open("cpu_timing.txt", "a")
  file.write("0.6 GeV - ")
  file.close()
  os.system("python run_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=0GeV6 --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9 --save-model")
  print("0.6 GeV complete")

  file = open("cpu_timing.txt", "a")
  file.write("0.5 GeV - ")
  file.close()
  os.system("python run_interaction_network.py --batch-size=1 --test-batch-size=1 --epochs=1 --pt=0GeV5 --no-cuda --construction=heptrkx_plus --lr=0.005 --gamma=0.9 --save-model")
  print("0.5 GeV complete")

else:
  print("no gpu code yet")
"""
