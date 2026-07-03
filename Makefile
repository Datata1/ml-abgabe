.PHONY: help figures present notebook experiment cache cache-replay test

help:  ## Diese Übersicht anzeigen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

figures:  ## Alle Grafiken als PNG -> reports/figures/ (fertig für die Folien)
	uv run python scripts/make_figures.py

present:  ## Notebook im Präsentationsmodus starten (read-only App-Ansicht)
	uv run marimo run notebooks/jd_praesentation.py

notebook:  ## Notebook zum Bearbeiten öffnen (Code + Zellen)
	uv run marimo edit notebooks/jd_praesentation.py

experiment:  ## Reproduzierbare Zahlen -> reports/results.csv
	uv run python scripts/run_experiment.py

cache:  ## VOR dem Vortrag: LLM-Routing cachen -> reports/llm_cache.json (offline-sicher, Provider nötig)
	uv run python scripts/precompute.py

cache-replay:  ## Gecachte LLM-Wahl auf dem aktuellen Split neu scoren (KEIN Provider nötig)
	uv run python scripts/precompute.py --replay

test:  ## Schnelle, datenfreie Tests
	uv run pytest -m "not data"
