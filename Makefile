.PHONY: test rank-sweep

test:
	python -m pytest -q

rank-sweep:
	python -m qlora_lab.experiments --output reports/rank_sweep.json
