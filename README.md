# Vehicle-Scale-LLMs-Integrating-Low-Rank-Residuals-and-4-Bit-Quantization-for-In-Vehicle-AI

```

dac-q4-its/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ CONTRIBUTING.md
├─ CHANGELOG.md
├─ pyproject.toml
├─ setup.cfg
├─ Makefile
├─ .gitignore
├─ .gitattributes
├─ .env.example
├─ .github/
│  ├─ ISSUE_TEMPLATE/
│  │  ├─ bug_report.md
│  │  └─ feature_request.md
│  └─ workflows/
│     ├─ ci.yaml                 # lint + unit tests
│     └─ docs.yaml               # build docs site (optional)
│
├─ docker/
│  ├─ Dockerfile.cuda11.8.trt8.6.1.6     # Ubuntu 18.04/CUDA/TensorRT pinned as in paper
│  ├─ Dockerfile.cpu.dev                 # dev-only CPU image
│  ├─ entrypoint.sh
│  └─ requirements.lock.txt              # fully pinned wheels for Docker image
│
├─ env/
│  ├─ environment.yml                    # conda env (CUDA/TRT notes in docs)
│  └─ requirements.txt                   # pip fallback
│
├─ configs/
│  ├─ model/
│  │  ├─ llama-1b.yml                    # hidden_size, n_layers, vocab_size…
│  │  └─ tokenizer.json                  # placeholder; script fetches at build
│  ├─ quant/
│  │  ├─ int4_dynamic.yml                # per-row scales; int range; rounding
│  │  └─ int4_static.yml                 # optional baseline
│  ├─ adapter/
│  │  └─ rank8_zero_init.yml             # rank, init, storage dtype
│  ├─ hw/
│  │  ├─ jetson_xavier_nx.yml            # SM version, TRT flags, workspace
│  │  └─ server_a100.yml
│  ├─ eval/
│  │  ├─ nln.yml
│  │  ├─ crp.yml
│  │  └─ tf.yml
│  └─ seeds.json                         # fixed seeds used for all runs
│
├─ scripts/
│  ├─ 00_fetch_base_model.sh             # pulls LLaMA-derived weights via user license
│  ├─ 01_prepare_calibration.py          # builds 25k-line corpus from recipes
│  ├─ 02_export_onnx.py                  # FP16→ONNX export (per-projection)
│  ├─ 03_quantize_int4.py                # produce INT4 weights + scale vectors
│  ├─ 04_compute_eigenspaces.py          # layerwise SVD on activations
│  ├─ 05_inject_adapters.py              # build U,D from ΔW projections
│  ├─ 06_build_trt_engine.py             # ONNX→TensorRT with custom plugins
│  ├─ 07_run_inference.py                # text generation CLI (FP16/8bit/4bit+DAC)
│  ├─ 08_eval_nln.py                     # Exact Match, SOFT F1
│  ├─ 09_eval_crp.py                     # BLEU-4 + HCS export
│  ├─ 10_eval_tf.py                      # ROUGE-L + MAE
│  ├─ 11_make_figures.py                 # reproduce Figures (incl. Table 8 bar chart)
│  ├─ 12_benchmark_latency.py            # per-token latency & memory profiling
│  └─ 99_pack_release_artifacts.py       # tar.gz models + checksums
│
├─ src/
│  └─ dac_q4_its/
│     ├─ __init__.py
│     ├─ logging.py
│     ├─ utils/
│     │  ├─ io.py                        # safeload/save, checksum, huggingface hub utils
│     │  ├─ seeds.py                     # global seeding utilities
│     │  ├─ metrics.py                   # EM, SOFT F1, BLEU-4, ROUGE-L, MAE, CIs
│     │  ├─ human_eval.py                # HCS parsing, IAA (Fleiss’ κ)
│     │  └─ profiling.py                 # memory & latency profilers
│     ├─ data/
│     │  ├─ recipes/
│     │  │  ├─ talk2nav_subset.yaml      # exact IDs + splits used
│     │  │  ├─ traffic_reports.yaml      # source URLs/formats; scrapers
│     │  │  └─ synthetic_longtail.yaml   # grammars/templates/rules
│     │  ├─ build_corpus.py              # uses recipes → 25k lines
│     │  └─ tokenization.py
│     ├─ modeling/
│     │  ├─ loader.py                    # HF-style model + tokenizer loader
│     │  ├─ export_onnx.py               # stable export, per-op checks
│     │  └─ generation.py                # greedy/beam; streaming
│     ├─ quantization/
│     │  ├─ int4_dynamic.py              # per-row scales; pack/unpack
│     │  ├─ int4_static.py               # baseline
│     │  └─ perturbation.py              # ΔW = Wfp16 − dequant(Wint4)
│     ├─ adapters/
│     │  ├─ eigenspace.py                # covariance, randomized SVD
│     │  ├─ build_adapters.py            # U, D construction; zero-init handling
│     │  └─ inject.py                    # graph surgery (ONNX) & runtime add
│     ├─ runtimes/
│     │  ├─ onnxrt_backend.py            # CPU/GPU ONNXRuntime fallback
│     │  ├─ tensorrt_backend.py          # TRT engine build + execution
│     │  └─ plugins/
│     │     ├─ int4_matmul_plugin.cpp    # custom INT4×FP16 GEMM (TensorRT)
│     │     ├─ CMakeLists.txt
│     │     └─ build.sh
│     └─ evaluation/
│        ├─ nln.py
│        ├─ crp.py
│        └─ tf.py
│
├─ checkpoints/                          # not stored in repo; .gitignore’d
│  ├─ README.md                          # how to download & verify checksums
│  ├─ base/                              # symlink or download target for base weights
│  └─ dac_q4_its/
│     ├─ llama1b_int4_dyn/
│     │  ├─ layer_*/W_qint4.bin          # packed INT4 weights per projection
│     │  ├─ layer_*/scales.fp16.bin
│     │  └─ adapters/
│     │     └─ layer_*/U.fp32.bin, D.fp32.bin
│     └─ model_index.json                # pointers + metadata
│
├─ data/                                  # small samples only; raw corpora .gitignore’d
│  ├─ samples/
│  │  ├─ nln_samples.jsonl
│  │  ├─ crp_samples.jsonl
│  │  └─ tf_samples.jsonl
│  └─ README.md
│
├─ notebooks/
│  ├─ 01_quickstart.ipynb                # load checkpoint & generate
│  ├─ 02_build_corpus.ipynb
│  ├─ 03_quantize_and_inject.ipynb
│  └─ 04_evaluate_tasks.ipynb
│
├─ tests/
│  ├─ test_quantization.py               # INT4 pack/unpack, error bounds
│  ├─ test_adapters.py                   # U,D shapes; reconstruction accuracy
│  ├─ test_eigenspace.py                 # SVD stability; top-k coverage
│  ├─ test_tensorrt_backend.py           # engine build sanity & kernel outputs
│  ├─ test_metrics.py
│  └─ test_reproducibility.py            # seeds, determinism smoke tests
│
└─ docs/
   ├─ index.md
   ├─ quickstart.md
   ├─ reproducibility_checklist.md       # items the reviewer requested
   ├─ environment.md                     # exact versions of Ubuntu/CUDA/TRT/PyTorch
   ├─ tensorrt_flags_xavier_nx.md        # plugins, calib, workspace, FP16/INT8 toggles
   ├─ human_eval_guidelines.md           # HCS rubric + annotator training
   ├─ calibration_corpus.md              # reconstruction steps & scripts
   ├─ figures/
   │  ├─ figure1_architecture.svg        # updated with INT4/FP16/FP32 labels
   │  ├─ table8_bar_chart.png            # dynamic vs static quantization
   │  └─ …
   └─ results_reproduction.md            # “run-these-commands” to rebuild tables 1–6



```
