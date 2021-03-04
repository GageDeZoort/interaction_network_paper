import os
import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--pt', type=str, default='2GeV',
                    help='Cutoff pt value in GeV (default: 2GeV)')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for inference')
parser.add_argument('--construction', type=str, default='heptrk_classic',
                    help='graph construction method')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='perform inference on GPU')
args = parser.parse_args()

file = open("timeit_interaction_network.sh", "r")
lines = file.readlines()
lines[1] = 'PT="{}"'.format(args.pt)
lines[2] = "BATCHSIZE={}".format(args.batch_size)
lines[4] = 'CONSTRUCTION="{}"'.format(args.construction)
lines[5] = "CUDA={}".format(1 if args.gpu else 0)
file = open("timeit_interaction_network.sh", "w")
file.writelines(lines)
file.close()

if(args.gpu):
  filename = "gpu_timing_{}_{}.txt".format(args.construction, args.pt)
  title = "CPU+GPU inference timing \n"
  device = "CPU+GPU"
else:
  filename = "cpu_timing_{}_{}.txt".format(args.construction, args.pt)
  title = "CPU inference timing \n"
  device = "CPU"

outfile = open(filename, "w")
outfile.write(title)
outfile.close()

for x in range(0,100):
  file = open("timeit_interaction_network.sh", "r")
  lines = file.readlines()
  lines[3] = 'GRAPHBATCHNUM="{}"'.format(x)
  file = open("timeit_interaction_network.sh", "w")
  file.writelines(lines)
  file.close()

  cmd = subprocess.Popen(['bash', 'timeit_interaction_network.sh'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
  stdout,stderr = cmd.communicate()
  outfile = open(filename, "a")
  outfile.write("\n")
  outfile.write("{} \n".format(stdout))
  outfile.close()

acc = []
with open(filename, "r") as f:
  for line in f.readlines():
    if '100 loops, best of 5:' in line:
      acc.append(int(line[22:26].strip()))

avg = 0
for x in len(acc):
  avg += acc[x]

avg /= len(acc)

print("average {} inference time(pt cut = {}, model = {}) = {}".format(device, args.pt, args.construction)) 
