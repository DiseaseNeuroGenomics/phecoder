SHELL := /bin/bash

cpu_queue?=premium
time?=12:00
memory?=6000
cores?=4

#### Submit GPU job (Minerva)
gpu?=a10080g
gpu_queue?=gpu
n_gpus?=1
submit-gpu:
	bsub -J "$(script) $(args)" \
		 -P acc_roussp01a \
		 -q $(gpu_queue) \
		 -n $(cores) \
		 -W $(time) \
		 -R rusage[mem=$(memory)] \
		 -R span[hosts=1] \
		 -R $(gpu) \
		 -gpu num=$(n_gpus) \
		 -o ./minerva/logs/gpu_%J.stdout \
		 -eo ./minerva/logs/gpu_%J.stderr \
		 -L /bin/bash \
		 ./minerva/gpu_job.sh "$(script)" $(args)

#### Remove local Minerva logs
clear-logs:
	rm -rf ./minerva/logs/*