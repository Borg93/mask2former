help:
	@echo Script to automate parts for this projects.

clean: clean_cache clean_runs

clean_cache: 
	rm -r ./cache/*

clean_runs: 
	rm -r ./runs/*

# Added pytest, linter

vega_nodes:
	srun -p gpu --gres=gpu:4 --nodes=1 --time=12:00:00 --mem=48GB --cpus-per-gpu=4 --pty bash

vega_drop_nodes:
	scancel -u euerikl

ready_test:
	pip install -e .

storage:
	du -sh