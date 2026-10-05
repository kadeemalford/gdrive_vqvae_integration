PYTHON ?= .venv/bin/python
CONFIG ?= config/spatial_range_pilot.json
RAW_DIR ?= data/raw
SAMPLES_DIR ?= data/samples
INFERENCE_RESULTS_DIR ?= results/inference_results
INFERENCE_WORK_DIR ?= $(INFERENCE_RESULTS_DIR)/.work
INFERENCE_RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
INFERENCE_RUN_DIR := $(INFERENCE_RESULTS_DIR)/$(INFERENCE_RUN_ID)
INFERENCE_CHECKPOINT ?= models/febus_finetune_base/best.pt
INFERENCE_DEVICE ?= cuda
FINETUNE_CONFIG ?= config/febus_finetune.json
FINETUNE_FOLDS ?= 4
FINETUNE_DIR ?= data/finetune
FINETUNE_TEST_FOLD ?= 0
FINETUNE_VAL_FOLD ?= 1
FINETUNE_REPO_ROOT ?= /home/murilo-barroso/source/smartgrids
FINETUNE_CHECKPOINT ?= models/febus_finetune_base/best.pt
FINETUNE_EPOCHS ?= 10
FINETUNE_LR ?= 0.00001
FINETUNE_MAX_TRAIN_SAMPLES ?= 0
FINETUNE_MAX_VAL_SAMPLES ?= 0
FINETUNE_EARLY_STOPPING_PATIENCE ?= 3
FINETUNE_EARLY_STOPPING_MIN_DELTA ?= 0.0005
FINETUNE_RESUME ?=
FINETUNE_SAMPLING ?= window
FINETUNE_STEPS_PER_EPOCH ?= 0
FINETUNE_OUTPUT_DIR ?= results/fine_tune_results/fold_$(FINETUNE_TEST_FOLD)

# Local execution settings. Command-line assignments take precedence.
-include .env

.DEFAULT_GOAL := help
.PHONY: help download inventory extract reconstruct metrics plot acquire process analyze all finetune-inventory finetune-folds finetune-download finetune-prepare finetune-train finetune-evaluate finetune-run

help:
	@printf '%s\n' 'DAS VQ-VAE pipeline'
	@printf '%s\n' ''
	@printf '%s\n' 'make download     Download HDF5 files from Drive'
	@printf '%s\n' 'make inventory    Inspect datasets and FEBUS metadata'
	@printf '%s\n' 'make extract      Extract configured FEBUS windows'
	@printf '%s\n' 'make reconstruct  Encode and decode windows with VQ-VAE'
	@printf '%s\n' 'make metrics      Calculate fidelity metrics'
	@printf '%s\n' 'make plot         Generate diagnostic figures'
	@printf '%s\n' 'make acquire      Run download and inventory'
	@printf '%s\n' 'make process      Run extract and reconstruct'
	@printf '%s\n' 'make analyze      Run metrics and plot'
	@printf '%s\n' 'make all          Run the complete pipeline'
	@printf '%s\n' 'make finetune-inventory  Build raw FEBUS manifest from Drive'
	@printf '%s\n' 'make finetune-folds      Assign conservative configuration folds'
	@printf '%s\n' 'make finetune-download   Download raw HDF5 files listed in the folds'
	@printf '%s\n' 'make finetune-prepare    Extract train/validation/test windows for one fold'
	@printf '%s\n' 'make finetune-train      Fine-tune best.pt for the selected fold'
	@printf '%s\n' 'make finetune-evaluate   Evaluate the fine-tuned checkpoint on the held-out fold'
	@printf '%s\n' 'make finetune-run        Train, then evaluate on the held-out fold'
	@printf '%s\n' ''
	@printf '%s\n' 'Local defaults are loaded from .env; start with .env.example.'
	@printf '%s\n' 'Override the run configuration with: make all CONFIG=config/pilot.json'
	@printf '%s\n' 'Inference checkpoint: INFERENCE_CHECKPOINT=...; analyze creates a timestamped run directory.'

download:
	@$(PYTHON) acquisition/download_selected_files.py --config $(CONFIG) --output-dir $(RAW_DIR)

inventory:
	@$(PYTHON) acquisition/inventory_hdf5.py --config $(CONFIG) --raw-dir $(RAW_DIR)

extract:
	@$(PYTHON) acquisition/extract_febus_windows.py --config $(CONFIG) --raw-dir $(RAW_DIR) --output-dir $(SAMPLES_DIR)

reconstruct:
	@$(PYTHON) compression/encode_decode_samples.py --config $(CONFIG) --checkpoint $(INFERENCE_CHECKPOINT) --device $(INFERENCE_DEVICE) --samples-dir $(SAMPLES_DIR) --output-dir $(INFERENCE_WORK_DIR)/reconstructions --metrics-dir $(INFERENCE_WORK_DIR)/metrics --summary-output $(INFERENCE_WORK_DIR)/metrics/reconstruction_summary.csv

metrics:
	@$(PYTHON) compression/compute_fidelity_metrics.py --recon-dir $(INFERENCE_WORK_DIR)/reconstructions --output $(INFERENCE_WORK_DIR)/metrics/fidelity_metrics.csv

plot:
	@$(PYTHON) presentation/plot_results.py --config $(CONFIG) --recon-dir $(INFERENCE_WORK_DIR)/reconstructions --output-dir $(INFERENCE_WORK_DIR)/figures --summary $(INFERENCE_WORK_DIR)/metrics/reconstruction_summary.csv

acquire: download inventory

process: extract reconstruct

analyze: metrics plot
	@$(PYTHON) presentation/finalize_inference_run.py --work-dir $(INFERENCE_WORK_DIR) --run-dir $(INFERENCE_RUN_DIR) --config $(CONFIG) --checkpoint $(INFERENCE_CHECKPOINT) --env .env

all: acquire process analyze

finetune-inventory:
	@$(PYTHON) finetuning/inventory_febus_drive.py --config $(FINETUNE_CONFIG)

finetune-folds:
	@$(PYTHON) finetuning/build_configuration_folds.py --folds $(FINETUNE_FOLDS)

finetune-download:
	@$(PYTHON) finetuning/download_febus_raw.py --output-dir $(FINETUNE_DIR)/raw

finetune-prepare:
	@$(PYTHON) finetuning/prepare_finetune_samples.py --raw-dir $(FINETUNE_DIR)/raw --output-dir $(FINETUNE_DIR)/fold_$(FINETUNE_TEST_FOLD) --test-fold $(FINETUNE_TEST_FOLD) --val-fold $(FINETUNE_VAL_FOLD)

finetune-train:
	@$(PYTHON) finetuning/finetune_vqvae.py --repo-root $(FINETUNE_REPO_ROOT) --checkpoint $(FINETUNE_CHECKPOINT) --samples-dir $(FINETUNE_DIR)/fold_$(FINETUNE_TEST_FOLD) --output-dir $(FINETUNE_OUTPUT_DIR) --epochs $(FINETUNE_EPOCHS) --learning-rate $(FINETUNE_LR) --max-train-samples $(FINETUNE_MAX_TRAIN_SAMPLES) --max-val-samples $(FINETUNE_MAX_VAL_SAMPLES) --early-stopping-patience $(FINETUNE_EARLY_STOPPING_PATIENCE) --early-stopping-min-delta $(FINETUNE_EARLY_STOPPING_MIN_DELTA) --sampling $(FINETUNE_SAMPLING) --steps-per-epoch $(FINETUNE_STEPS_PER_EPOCH) $(if $(FINETUNE_RESUME),--resume $(FINETUNE_RESUME))

finetune-evaluate:
	@$(PYTHON) finetuning/evaluate_checkpoints.py --repo-root $(FINETUNE_REPO_ROOT) --samples-dir $(FINETUNE_DIR)/fold_$(FINETUNE_TEST_FOLD)/test --checkpoint base=$(FINETUNE_CHECKPOINT) --checkpoint finetuned=$(FINETUNE_OUTPUT_DIR)/best.pt --output-dir $(FINETUNE_OUTPUT_DIR)/evaluation

finetune-run: finetune-train finetune-evaluate
