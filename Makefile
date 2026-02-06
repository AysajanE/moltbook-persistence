SHELL := /bin/bash

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff

.PHONY: help venv install lint format paper clean-paper arxiv-bundle

help:
	@echo "Targets:"
	@echo "  make venv         Create .venv"
	@echo "  make install      Install deps (incl dev)"
	@echo "  make format       Auto-format (ruff)"
	@echo "  make lint         Lint (ruff)"
	@echo "  make paper        Build paper PDF (latexmk)"
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

clean-paper:
	cd paper && latexmk -C

arxiv-bundle:
	"$(PY)" scripts/build_arxiv_bundle.py --paper-dir paper --out-root outputs/arxiv --check
