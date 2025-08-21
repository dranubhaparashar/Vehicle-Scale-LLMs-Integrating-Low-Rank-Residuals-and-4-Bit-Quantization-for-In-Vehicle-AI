.PHONY: all corpus export quant eigs adapters run eval

all: corpus export quant eigs adapters run eval

corpus:
	python scripts/01_prepare_calibration.py

export:
	python scripts/02_export_onnx.py

quant:
	python scripts/03_quantize_int4.py

eigs:
	python scripts/04_compute_eigenspaces.py

adapters:
	python scripts/05_inject_adapters.py

run:
	python scripts/07_run_inference.py --prompt "navigate to airport avoiding tolls"

eval:
	python scripts/08_eval_nln.py
