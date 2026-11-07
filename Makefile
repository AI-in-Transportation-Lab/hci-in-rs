.PHONY: install run list clean ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08 ch09 ch10 ch11 ch12

install:   ## create .venv and install dependencies (needs uv)
	uv sync

run:       ## run every case study
	uv run python scripts/run_all.py

list:
	uv run python scripts/run_all.py --list

ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08 ch09 ch10 ch11 ch12:
	uv run python scripts/run_all.py --chapter $(subst ch,,$@)

clean:     ## remove generated figures and caches
	find outputs -mindepth 1 ! -name .gitkeep -delete
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
