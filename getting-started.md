---
title: Getting started
nav_order: 2
permalink: /getting-started/
---

# Getting started
{: .no_toc }

1. TOC
{:toc}

## Install

You need Python 3.10 or newer. [uv](https://docs.astral.sh/uv/) is the easiest way to get a matching environment:

```bash
git clone https://github.com/AI-in-Transportation-Lab/hci-in-rs.git
cd hci-in-rs
uv sync                                   # creates .venv and installs dependencies
uv run python scripts/run_all.py          # run every case study
```

Using pip instead:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
```

## Running a subset

```bash
python scripts/run_all.py --list              # show what would run
python scripts/run_all.py --chapter 6         # one chapter
python scripts/run_all.py -c 3 -k 2           # chapter 3, case study 2
make ch08                                     # same thing via make
```

The runner executes each script with `outputs/chNN/` as its working directory, so figures the listings write with relative paths land there and never in the source tree. You can also run a script directly; output then goes to your current directory.

## Repository layout

```
hci-in-rs/
├── chapters/chNN_<topic>/cs<N>_<topic>.py   # one script per case study
├── scripts/run_all.py
├── docs/figures/                            # reference figures from the book
└── outputs/                                 # generated figures (git-ignored)
```

## Reproducibility notes

- All data are synthetic and generated with fixed seeds. Numbers can still differ slightly across library versions and platforms. `uv.lock` pins the versions used to check these scripts.
- The synthetic data illustrate methods. They are not agency or manufacturer data, and results should not be read as findings about real drivers, vehicles, or roadways.
- The scripts are the book's listings verbatim; only a short docstring header was added to each.

## Previewing this site locally

```bash
bundle install
bundle exec jekyll serve    # http://localhost:4000/hci-in-rs/
```
