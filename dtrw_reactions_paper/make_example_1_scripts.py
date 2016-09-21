#!/usr/local/bin/python3

launch_script_name = "launch_example_1.sh"
launch_script_file = open(launch_script_name, 'w')

Ls = [5.0, 10.0, 20.0, 50.0]
dXs = [0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
dXs = [0.01]
ks = [1.0, 10.0]
alphas = [0.9]

for alpha in alphas:
    for k in ks:
        for L in Ls:
            for dX in dXs:

                file_name = 'scripts/example_1_script_{0}_{1}_{2}_{3}.pbs'.format(alpha, k, L, dX)
                launch_script_file.write('qsub ' + file_name + '\n') 
                
                out_file = open(file_name, 'w')
                out_file.write('#!/bin/bash\n')
                out_file.write('\n') 
                out_file.write('#PBS -l nodes=1:ppn=1\n')
                out_file.write('#PBS -l vmem=16gb\n')
                out_file.write('#PBS -l walltime=48:00:00\n')
                out_file.write('#PBS -m ae\n')
                out_file.write('#PBS -M james.ashton.nichols@gmail.com\n')
                out_file.write('\n')
                out_file.write('cd /home/z3180058/projects/dtrw/dtrw_reactions_paper/\n')
                out_file.write('./Example_1.py {0} {1} {2} {3} {4} \n'.format(alpha, 2.0, L, dX, k))
                out_file.close()

