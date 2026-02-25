SHELL := /bin/bash

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff

.PHONY: help venv install lint format paper paper-arxiv paper-arxiv-appendix paper-ejor clean-paper arxiv-bundle

help:
	@echo "Targets:"
	@echo "  make venv         Create .venv"
	@echo "  make install      Install deps (incl dev)"
	@echo "  make format       Auto-format (ruff)"
	@echo "  make lint         Lint (ruff)"
	@echo "  make paper        Build arXiv manuscript PDF (paper/main.tex)"
	@echo "  make paper-arxiv  Build arXiv manuscript PDF (paper/main_arxiv.tex)"
	@echo "  make paper-arxiv-appendix  Build arXiv online appendix PDF (paper/main_arxiv_appendix.tex)"
	@echo "  make paper-ejor   Build EJOR manuscript PDF (paper/main_ejor.tex)"
	@echo "  make clean-paper  Clean paper build artifacts"
	@echo "  make arxiv-bundle Build timestamped arXiv source tarball"

venv:
	python -m venv "$(VENV)"

install: venv
	"$(PIP)" install -r requirements.txt -r requirements-dev.txt

format:
	"$(RUFF)" format .

lint:
	"$(RUFF)" check .

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-arxiv:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main_arxiv.tex

paper-arxiv-appendix:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main_arxiv_appendix.tex

paper-ejor:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main_ejor.tex

clean-paper:
	cd paper && latexmk -C main.tex
	cd paper && latexmk -C main_arxiv.tex
	cd paper && latexmk -C main_arxiv_appendix.tex
	cd paper && latexmk -C main_ejor.tex

arxiv-bundle:
	"$(PY)" scripts/build_arxiv_bundle.py --paper-dir paper --out-root outputs/arxiv --check
