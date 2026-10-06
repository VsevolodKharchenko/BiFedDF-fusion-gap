# Investigating the Fusion Gap in Bidirectional Federated Distillation Systems

Bachelor's thesis project by **Vsevolod Kharchenko**, B.Sc. Artificial Intelligence, Institute of Pervasive Computing, Johannes Kepler University Linz (July 2026).

- Supervisor: Univ.-Prof. Dr. Alois Ferscha
- Assistant supervisor: Prof. MSc. Muhammad Waleed Khan, MS

This repository holds the thesis, the code and notebooks for its experiments, and the earlier practical work the thesis builds on.

## Thesis summary

Federated learning (FL) trains models on decentralized user data without collecting it. In practice it runs into model heterogeneity and non-IID local data. Frameworks such as **FedDF** (Federated Distillation Fusion) address this by distilling the clients' knowledge into the server model on a proxy dataset. At high levels of data heterogeneity, however, accuracy still degrades through catastrophic forgetting. Client models also stay relatively stable while the server model suffers. The thesis calls this difference the **fusion gap**.

The thesis studies the fusion gap empirically under different levels of data heterogeneity, using Dirichlet-distributed client splits of CIFAR-10. It compares two setups:

- **FedDF**: clients train locally and the server fuses their outputs by ensemble distillation.
- **BiFedDF**: FedDF with an added bidirectional distillation component, where knowledge also flows from the server back to the clients.

Main findings:

- BiFedDF trades server accuracy for much better client accuracy. The bidirectional distillation acts as a regularizer on the clients, which reduces catastrophic forgetting of classes that are underrepresented in the local data. It also sharply reduces the gap between server and client accuracy.
- The same regularization lowers the diversity of the client models. This makes the client ensemble a weaker teacher, so server accuracy drops and the fusion gap grows, especially as heterogeneity increases.
- Bidirectional distillation therefore trades global server accuracy against local client stability.

The full text is in [`Fed_comp_bachelor (18).pdf`](Fed_comp_bachelor%20(18).pdf). Its chapters are:

1. Introduction
2. Background and Related Work (data heterogeneity, knowledge distillation, federated learning, bidirectional distillation)
3. Methodology and Experimental Setup
4. Experiment Results (server vs. client performance, the fusion gap, catastrophic forgetting, loss of ensemble diversity)
5. Conclusions and Future Work

## Repository contents

### Thesis experiments

| File | Description |
|------|-------------|
| `Fed_comp_bachelor (18).pdf` | The thesis. |
| `part3.ipynb` | Main experiment notebook. It implements FedAvg, FedDF and BiFedDF (bidirectional) in PyTorch, with Dirichlet-based non-IID client partitioning. It uses a ResNet-18 teacher and the smaller ResNet-14 student. It also contains the experiment runs over several seeds and the analysis and plots: server vs. client accuracy per round, the fusion gap, per-class accuracy heatmaps, pairwise client agreement, and ensemble accuracy. |
| `part1.ipynb` | Model preparation. It trains the ResNet-18 baseline on CIFAR-10 and builds the ResNet-14 student (ResNet-18 without the 4th stage, about 2.8M parameters vs. 11.2M) by knowledge distillation. |
| `resnet18_best.pth` | Trained ResNet-18 baseline weights (teacher). |
| `big_student_distilled_best.pth` | Distilled student weights. |
| `baseline_params.json`, `student_resnet14_params.json` | Parameter counts of the two models. |
| `stage1_results.csv` | Size, accuracy and latency comparison of the baseline and ResNet-14. |

Baseline vs. student on CIFAR-10:

| Model | Size (MB) | Accuracy (%) | Latency (ms) |
|-------|-----------|--------------|--------------|
| ResNet-18 (baseline) | 41.5 | 87.0 | 156.1 |
| ResNet-14 (student) | 10.4 | 84.6 | 12.6 |

`part1 copy.ipynb` and `part3backup0.ipynb` are earlier backup versions of the notebooks.

### Earlier practical work (`practicalwork_untouched/`)

This is the earlier practical work, kept unchanged, that the thesis grew out of. It is the paper *Model Compression for Resource-Constrained Federated Learning Using Knowledge Distillation* (February 2026). It builds a multi-stage compression pipeline for FL on edge devices:

| File | Stage |
|------|-------|
| `part1.ipynb` | **Stage 1: model compression on CIFAR-10.** Trains a ResNet-18 baseline and compares response-based knowledge distillation, magnitude pruning (30/50/70%) and 8-bit post-training quantization. |
| `part2.ipynb` | **Stage 2: edge deployment.** Converts models to TensorFlow Lite, tests INT8/INT4 quantization, and measures CPU latency, memory footprint and energy estimates. |
| `part3.ipynb` | **Stage 3: federated learning.** Simulates a heterogeneous FL setup with adaptively pruned and quantized clients, comparing FedAvg with FedDF for bandwidth savings. |
| `practical.pdf` | Written report for the practical work. |

## Tech stack

Python, PyTorch, torchvision, NumPy, pandas, matplotlib, seaborn and tqdm. The practical work's Stage 2 also uses TensorFlow, `ai_edge_torch` and `torchao`.

## Running the notebooks

1. Install the dependencies above (a CUDA GPU is recommended for training).
2. The CIFAR-10 dataset is not included in the repository. torchvision downloads it into a local `data/` folder when a notebook asks for it.
3. Run `part1.ipynb` first to produce the teacher and student models. Pretrained weights are included, so you can go straight to `part3.ipynb`.

## Notes

- Visualizations in the notebooks and the thesis were made with LLM assistance. Gemini was also used for structural planning and proofreading. The research methodology, experiments and analysis are the author's own, as stated in the thesis's declaration.
