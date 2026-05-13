# ──────────────────────────────────────────────────────────────────────────────
# Intrusion Forge — Experiment Runner
#
# Usage:
#   make prepare  DATA=cic_2018_v2 NAME=my_exp
#   make classify DATA=cic_2018_v2 NAME=my_exp
#   make analyze  DATA=cic_2018_v2 NAME=my_exp
#   make run      DATA=cic_2018_v2 NAME=my_exp    # all three phases
#   make all      NAME=my_exp                      # all datasets
# ──────────────────────────────────────────────────────────────────────────────

PYTHON     := venv/bin/python
EXPERIMENT ?= supervised
DATA       ?= cic_2018_v2
NAME       ?= exp
SEED       ?= 42
MODEL      ?= tabular_classifier
DISTANCE   ?= cosine

DATASET_MODELS := \
    nb15_v2:tabular_classifier \
    bot_iot_v2:tabular_classifier \
    cic_2018_v2:tabular_classifier \
    ton_iot_v2:tabular_classifier \
    bank_marketing:tabular_classifier \
    covertype:numerical_classifier \
    letter_recognition:numerical_classifier \
    statlog_landsat_satellite:numerical_classifier \
    thyroid_disease:numerical_classifier

HYDRA := experiment=$(EXPERIMENT) data=$(DATA) name=$(NAME) seed=$(SEED) \
         model=$(MODEL) \
         complexity.distance=$(DISTANCE) clustering.distance=$(DISTANCE)
TB_LOGDIR  := resources/experiments/$(NAME)/$(DATA)_$(SEED)/tb

.PHONY: prepare classify analyze run all generate tensorboard help

## prepare:            Step 1 — preprocess raw CSV → parquet splits  (DATA, NAME, SEED)
prepare:
	$(PYTHON) prepare_data.py $(HYDRA)

## classify:           Step 2 — train & evaluate classifier           (DATA, NAME, SEED)
classify:
	$(PYTHON) classify.py $(HYDRA)

## analyze:            Step 3 — post-hoc analysis                     (DATA, NAME, SEED)
analyze:
	$(PYTHON) analyze_data.py $(HYDRA)

## run:                Run all three steps for a single dataset        (DATA, NAME, SEED)
run: prepare classify analyze

## all:                Run all three steps for every dataset in DATASET_MODELS (NAME, SEED, DISTANCE)
all:
	@for entry in $(DATASET_MODELS); do \
		dataset=$${entry%%:*}; model=$${entry##*:}; \
		echo ""; \
		echo "══════════════════════════════════════════════"; \
		echo " Dataset: $$dataset  |  model=$$model  |  name=$(NAME)  seed=$(SEED)"; \
		echo "══════════════════════════════════════════════"; \
		$(MAKE) --no-print-directory run \
			DATA=$$dataset MODEL=$$model NAME=$(NAME) SEED=$(SEED) EXPERIMENT=$(EXPERIMENT) \
			DISTANCE=$(DISTANCE); \
	done
	@echo ""
	@echo "All datasets processed."

## generate:           Generate synthetic test dataset                 (ROWS)
generate:
	$(PYTHON) generate_synthetic.py $(if $(ROWS),--rows $(ROWS),)

## tensorboard:        Open TensorBoard for the current experiment     (DATA, NAME, SEED)
tensorboard:
	venv/bin/tensorboard --logdir $(TB_LOGDIR)

## help:               Show this help message
help:
	@echo "Usage: make <target> [DATA=<dataset>] [NAME=<name>] [SEED=<n>] [EXPERIMENT=<exp>] [MODEL=<model>] [DISTANCE=<dist>]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Defaults:  DATA=$(DATA)  NAME=$(NAME)  SEED=$(SEED)  EXPERIMENT=$(EXPERIMENT)  MODEL=$(MODEL)  DISTANCE=$(DISTANCE)"
	@echo "Datasets:  $(DATASET_MODELS)"
	@echo "TensorBoard logdir:  $(TB_LOGDIR)"
