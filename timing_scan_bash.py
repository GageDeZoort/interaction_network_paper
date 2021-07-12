import os
import subprocess
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=1, metavar='N',
                    help='input batch size for inference')
parser.add_argument('--construction', type=str, default='heptrkx_plus_pyg',
                    help='graph construction method')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='perform inference on GPU')
parser.add_argument('--graphs', type=int, default=5,
                    help='number of different graphs to average results over')
parser.add_argument('--loops', type=int, default=100,
                    help='number of executions to loop for timing test; see https://docs.python.org/3/library/timeit.html')
parser.add_argument('--repeat', type=int, default=5,
                    help='number of repetitions of execution loop to find best timing; see https://docs.python.org/3/library/timeit.html')
args = parser.parse_args()

pts = ['2GeV', '1GeV5', '1GeV', '0GeV9', '0GeV8', '0GeV7', '0GeV6', '0GeV5']

for pt in pts:
  file = open("timeit_interaction_network.sh", "r")
  lines = file.readlines()
  lines[1] = 'PT="{}"\n'.format(pt)
  lines[2] = "BATCHSIZE={}\n".format(args.batchsize)
  lines[4] = 'CONSTRUCTION="{}"\n'.format(args.construction)
  lines[5] = "CUDA={}\n".format(1 if args.gpu else 0)
  lines[6] = "NUMBER={}\n".format(args.loops)
  lines[7] = "REPEAT={}\n".format(args.repeat)
  file = open("timeit_interaction_network.sh", "w")
  file.writelines(lines)
  file.close()

  if(args.gpu):
    filename = "gpu_timing_{}_{}.txt".format(args.construction, pt)
    title = "GPU inference timing \n"
    device = "GPU"
  else:
    filename = "cpu_timing_{}_{}.txt".format(args.construction, pt)
    title = "CPU inference timing \n"
    device = "CPU"

  outfile = open(filename, "w")
  outfile.write(title)
  outfile.close()
  for x in range(0,5):
    file = open("timeit_interaction_network.sh", "r")
    lines = file.readlines()
    lines[3] = "GRAPHBATCHNUM={}\n".format(x)
    file = open("timeit_interaction_network.sh", "w")
    file.writelines(lines)
    file.close()

    cmd = subprocess.check_output(['bash', 'timeit_interaction_network.sh']).splitlines()
    for i, line in enumerate(cmd):
      if 'loops, best of' in str(line):
        print(str(line))
        outfile = open(filename, "a")
        outfile.write("\n")
        outfile.write("{} \n".format(line))
        outfile.close()
      else:
        continue

  acc = []
  with open(filename, "r") as f:
    for line in f.readlines():
      if 'loops, best of' in line:
        print("parse line recognized")
        value = float(line.split(' ')[5])
        units = line.split(' ')[6]
        if units=='usec':
          value = value*1e-3
        elif units=='sec':
          value = value*1e3
        acc.append(value)

  avg = np.mean(acc)
  std = np.std(acc)

  print("average {} inference time (pt cut = {}, model = {}) = {} +/- {} msec".format(device, pt, args.construction, avg, std)) 
  os.system("echo 'average {} inference time (pt cut = {}, model = {}) = {} +/- {} msec' >> {}".format(device, pt, args.construction, avg, std, filename))
