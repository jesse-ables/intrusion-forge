# ──────────────────────────────────────────────────────────────────────────────
# Intrusion Forge — Experiment Runner
#
# Usage:
#   make prepare           DATA=cic_2018_v2 NAME=my_exp
#   make classify          DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make ml-all            DATA=cic_2018_v2 NAME=my_exp     # every ML classifier
#   make dl-all            DATA=cic_2018_v2 NAME=my_exp     # every DL classifier
#   make complexity        DATA=cic_2018_v2 NAME=my_exp     # shared, dataset-level
#   make failure-classify  DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make render            DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make run               DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=tabular
#   make run-all           DATA=cic_2018_v2 NAME=my_exp      # one dataset, all compatible classifiers
#   make all               NAME=my_exp                       # all datasets, all compatible classifiers
#
# Add FORCE=1 to recompute cached shared stages (prepare, complexity).
# ──────────────────────────────────────────────────────────────────────────────

# Use venv if present; falls back to the active conda (or system) python otherwise.
# Override explicitly: make <target> PYTHON=python
PYTHON    ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)
STREAMLIT ?= $(if $(wildcard venv/bin/streamlit),venv/bin/streamlit,streamlit)
DATA       ?= cic_2018_v2
NAME       ?= exp_euc
SEED       ?= 42
CLASSIFIER ?= tabular
DISTANCE   ?= euclidean
FORCE      ?=

ML_CLASSIFIERS := \
    naive_bayes \
    logistic_regression \
    lda \
    knn \
    decision_tree \
    random_forest \
    hist_gradient_boosting \
    svm_rbf \
    xgboost

DL_CLASSIFIERS_MIXED     := tabular
DL_CLASSIFIERS_NUMERICAL := numerical

DATASET_FORMATS := \
    nb15_v2:mixed \
    bot_iot_v2:mixed \
    cic_2018_v2:mixed \
    ton_iot_v2:mixed \
    bank_marketing:mixed \
    covertype:numerical \
    letter_recognition:numerical \
    statlog_landsat_satellite:numerical \
    thyroid_disease:numerical

HYDRA := data=$(DATA) name=$(NAME) seed=$(SEED) classifier=$(CLASSIFIER) \
         complexity.distance=$(DISTANCE) clustering.distance=$(DISTANCE)
FORCE_FLAG := $(if $(FORCE),prepare.force=true complexity.force=true,)

.PHONY: prepare classify ml-all dl-all complexity failure-classify render run run-all all generate dashboard help

## prepare:            Step 1 — preprocess raw CSV → parquet splits           (DATA, NAME, SEED, FORCE)
prepare:
	PYTHONPATH=. $(PYTHON) pipelines/prepare_data.py $(HYDRA) $(FORCE_FLAG)

## classify:           Step 2 — train & evaluate one classifier (ML or DL)    (DATA, NAME, SEED, CLASSIFIER)
classify:
	PYTHONPATH=. $(PYTHON) pipelines/classify.py $(HYDRA)

## ml-all:             Step 2 — train & evaluate every ML classifier in turn  (DATA, NAME, SEED)
ml-all:
	@for clf in $(ML_CLASSIFIERS); do \
		echo ""; \
		echo "── ML classifier: $$clf ─────────────────────────────"; \
		$(MAKE) --no-print-directory classify \
			DATA=$(DATA) NAME=$(NAME) SEED=$(SEED) CLASSIFIER=$$clf \
			DISTANCE=$(DISTANCE) || exit 1; \
	done

## dl-all:             Step 2 — train & evaluate every compatible DL classifier  (DATA, NAME, SEED)
dl-all:
	@format=""; \
	for entry in $(DATASET_FORMATS); do \
		ds=$${entry%%:*}; fmt=$${entry##*:}; \
		if [ "$$ds" = "$(DATA)" ]; then format=$$fmt; break; fi; \
	done; \
	if [ -z "$$format" ]; then \
		echo "ERROR: DATA='$(DATA)' not found in DATASET_FORMATS."; exit 1; \
	fi; \
	if [ "$$format" = "mixed" ]; then \
		dl_list="$(DL_CLASSIFIERS_MIXED)"; \
	else \
		dl_list="$(DL_CLASSIFIERS_NUMERICAL)"; \
	fi; \
	for clf in $$dl_list; do \
		echo ""; \
		echo "── DL classifier: $$clf ─────────────────────────────"; \
		$(MAKE) --no-print-directory classify \
			DATA=$(DATA) NAME=$(NAME) SEED=$(SEED) CLASSIFIER=$$clf \
			DISTANCE=$(DISTANCE) || exit 1; \
	done

## complexity:         Step 3a — per-cluster complexity (shared, idempotent)  (DATA, NAME, SEED, FORCE)
complexity:
	PYTHONPATH=. $(PYTHON) pipelines/compute_complexity.py $(HYDRA) $(FORCE_FLAG)

## failure-classify:   Step 3b — RF to detect problematic clusters            (DATA, NAME, SEED, CLASSIFIER)
failure-classify: complexity
	PYTHONPATH=. $(PYTHON) pipelines/fit_failure_classifier.py $(HYDRA)

## render:             Step 4 — render plots from analysis artifacts          (DATA, NAME, SEED, CLASSIFIER)
render:
	PYTHONPATH=. $(PYTHON) pipelines/render_plots.py $(HYDRA)

## run:                Run all steps for a single (dataset, classifier)       (DATA, NAME, SEED, CLASSIFIER)
run: prepare classify failure-classify render

## run-all:            Run all steps for a single dataset, all compatible classifiers (DATA, NAME, SEED)
run-all:
	@format=""; \
	for entry in $(DATASET_FORMATS); do \
		ds=$${entry%%:*}; fmt=$${entry##*:}; \
		if [ "$$ds" = "$(DATA)" ]; then format=$$fmt; break; fi; \
	done; \
	if [ -z "$$format" ]; then \
		echo "ERROR: DATA='$(DATA)' not found in DATASET_FORMATS."; exit 1; \
	fi; \
	if [ "$$format" = "mixed" ]; then \
		dl_list="$(DL_CLASSIFIERS_MIXED)"; \
	else \
		dl_list="$(DL_CLASSIFIERS_NUMERICAL)"; \
	fi; \
	echo ""; \
	echo "══════════════════════════════════════════════"; \
	echo " Dataset: $(DATA)  |  format=$$format  |  name=$(NAME)  seed=$(SEED)"; \
	echo "══════════════════════════════════════════════"; \
	$(MAKE) --no-print-directory prepare \
		DATA=$(DATA) NAME=$(NAME) SEED=$(SEED) \
		DISTANCE=$(DISTANCE) $(FORCE_FLAG) || exit 1; \
	for clf in $(ML_CLASSIFIERS) $$dl_list; do \
		echo ""; \
		echo "── classifier: $$clf ─────────────────────────────"; \
		$(MAKE) --no-print-directory classify \
			DATA=$(DATA) NAME=$(NAME) SEED=$(SEED) CLASSIFIER=$$clf \
			DISTANCE=$(DISTANCE) || exit 1; \
		$(MAKE) --no-print-directory failure-classify \
			DATA=$(DATA) NAME=$(NAME) SEED=$(SEED) CLASSIFIER=$$clf \
			DISTANCE=$(DISTANCE) || exit 1; \
		$(MAKE) --no-print-directory render \
			DATA=$(DATA) NAME=$(NAME) SEED=$(SEED) CLASSIFIER=$$clf \
			DISTANCE=$(DISTANCE) || exit 1; \
	done

## all:                Run the full pipeline for every dataset (all ML + compatible DL classifiers) (NAME, SEED)
all:
	@for entry in $(DATASET_FORMATS); do \
		dataset=$${entry%%:*}; format=$${entry##*:}; \
		if [ "$$format" = "mixed" ]; then \
			dl_list="$(DL_CLASSIFIERS_MIXED)"; \
		else \
			dl_list="$(DL_CLASSIFIERS_NUMERICAL)"; \
		fi; \
		echo ""; \
		echo "══════════════════════════════════════════════"; \
		echo " Dataset: $$dataset  |  format=$$format  |  name=$(NAME)  seed=$(SEED)"; \
		echo "══════════════════════════════════════════════"; \
		$(MAKE) --no-print-directory prepare \
			DATA=$$dataset NAME=$(NAME) SEED=$(SEED) \
			DISTANCE=$(DISTANCE) $(FORCE_FLAG) || exit 1; \
		for clf in $(ML_CLASSIFIERS) $$dl_list; do \
			echo ""; \
			echo "── classifier: $$clf ─────────────────────────────"; \
			$(MAKE) --no-print-directory classify \
				DATA=$$dataset NAME=$(NAME) SEED=$(SEED) CLASSIFIER=$$clf \
				DISTANCE=$(DISTANCE) || exit 1; \
			$(MAKE) --no-print-directory failure-classify \
				DATA=$$dataset NAME=$(NAME) SEED=$(SEED) CLASSIFIER=$$clf \
				DISTANCE=$(DISTANCE) || exit 1; \
			$(MAKE) --no-print-directory render \
				DATA=$$dataset NAME=$(NAME) SEED=$(SEED) CLASSIFIER=$$clf \
				DISTANCE=$(DISTANCE) || exit 1; \
		done; \
	done
	@echo ""
	@echo "All datasets processed."

## generate:           Generate synthetic test dataset                        (ROWS)
generate:
	$(PYTHON) generate_synthetic.py $(if $(ROWS),--rows $(ROWS),)

## dashboard:          Open the experiment dashboard in browser
dashboard:
	$(STREAMLIT) run dashboard.py

## help:               Show this help message
help:
	@echo "Usage: make <target> [DATA=<dataset>] [NAME=<name>] [SEED=<n>] [CLASSIFIER=<name>] [DISTANCE=<dist>]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Defaults:  DATA=$(DATA)  NAME=$(NAME)  SEED=$(SEED)  CLASSIFIER=$(CLASSIFIER)  DISTANCE=$(DISTANCE)"
	@echo "Python:    $(PYTHON)  (override with PYTHON=)"
	@echo "ML classifiers:         $(ML_CLASSIFIERS)"
	@echo "DL classifiers (mixed): $(DL_CLASSIFIERS_MIXED)"
	@echo "DL classifiers (num):   $(DL_CLASSIFIERS_NUMERICAL)"
	@echo "Datasets (format):      $(DATASET_FORMATS)"
