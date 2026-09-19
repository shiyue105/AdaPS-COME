# AdaPS-COME: Adaptive Prediction-Set Reliability for Safe Test-Time Adaptation of Vision Transformers

AdaPS-COME is a reliability-aware framework for **Test-Time Adaptation (TTA)** of Vision Transformers (ViTs) under distribution shifts.

The method combines **adaptive prediction-set reliability** with a **COME-style conservative objective** to reduce the influence of ambiguous target samples during online adaptation.

## Overview

Test-Time Adaptation aims to adapt a source-trained model to an unlabeled target distribution without retraining the model from scratch.

However, target samples can vary substantially in ambiguity under distribution shifts. Highly ambiguous samples may have probability mass distributed across many classes, making them potentially unreliable signals for model adaptation.

AdaPS-COME addresses this issue by using the **prediction-set size** as a model-induced ambiguity proxy:

- Estimate batch-level uncertainty.
- Adaptively determine a probability-mass threshold.
- Construct a prediction set for each target sample.
- Use prediction-set size to estimate sample reliability.
- Assign lower adaptation weights to samples with larger prediction sets.
- Combine the reliability weighting with a COME-style conservative adaptation objective.

The resulting framework is designed to make online adaptation more reliability-aware under corruption shifts.

> **Important:** The prediction sets used in AdaPS-COME are a model-induced ambiguity proxy. They are **not conformal prediction sets** and do not provide finite-sample coverage guarantees.

---

## Method

### 1. Test-Time Adaptation

Given a source-trained model

$$
f_\theta
$$

and an unlabeled target stream

$$
\{x_i\}_{i=1}^{N},
$$

AdaPS-COME performs online adaptation using target samples without accessing their labels during adaptation.

In the ImageNet-C experiments, only the **affine parameters of LayerNorm layers** are updated, while the classifier and other model parameters remain frozen.

---

### 2. Adaptive Prediction-Set Threshold

For each target batch, AdaPS-COME first estimates the batch-level uncertainty.

The uncertainty is then mapped to an adaptive probability-mass threshold:

$$
q_t =
\mathrm{clip}
\left(
q_{\min}
+
(q_{\max}-q_{\min})U_t,
q_{\min},
q_{\max}
\right),
$$

where:

- $U_t$ is the uncertainty estimated from the current batch.
- $q_{\min}$ is the minimum threshold.
- $q_{\max}$ is the maximum threshold.

The default settings used in the experiments are:

```text
q_min = 0.90
q_max = 0.98

where:

- $U_t$ is the uncertainty estimated from the current batch.
- $q_{\min}$ is the minimum threshold.
- $q_{\max}$ is the maximum threshold.

The default settings used in the experiments are:

```text
q_min = 0.90
q_max = 0.98
```

A higher uncertainty level therefore produces a larger probability-mass threshold.

---

### 3. Prediction-Set Size

For each target sample, classes are ranked according to their predicted probabilities.

The prediction set is defined as the smallest set of top-ranked classes whose cumulative probability mass reaches the adaptive threshold $q_t$.

Let

$$
s_i(q_t)
$$

denote the resulting prediction-set size for sample $i$.

A larger prediction-set size indicates that the model distributes its probability mass over more classes, providing a model-induced indication of greater ambiguity.

---

### 4. Reliability Weighting

AdaPS-COME converts prediction-set size into a reliability weight:

$$
w_i = s_i^{-\gamma},
$$

where $\gamma$ controls the strength of the reliability weighting.

The default setting is:

```text
gamma = 0.5
```

Therefore:

- Small prediction set → higher reliability → larger adaptation weight.
- Large prediction set → greater ambiguity → smaller adaptation weight.

The reliability weights are detached from the computational graph during optimization.

---

### 5. COME-Based Conservative Objective

AdaPS-COME combines the reliability weighting with a COME-style conservative objective.

The final objective is based on a weighted combination of conventional predictive entropy and COME uncertainty:

$$
\mathcal{L}
=
\sum_i
w_i
\left[
(1-\lambda)H(p_i)
+
\lambda H_{\mathrm{COME}}(o_i)
\right]
$$

where:

- $p_i$ represents the model prediction.
- $o_i$ represents the subjective opinion used by COME.
- $H(p_i)$ is the predictive entropy.
- $H_{\mathrm{COME}}(o_i)$ is the COME-based uncertainty objective.
- $w_i$ is the reliability weight.
- $\lambda$ controls the contribution of the COME objective.

The default value is:

```text
lambda = 0.03
```

---

## Key Idea

The central idea of AdaPS-COME is:

```text
Target Sample
      │
      ▼
Model Prediction
      │
      ▼
Batch Uncertainty
      │
      ▼
Adaptive Threshold q_t
      │
      ▼
Prediction Set
      │
      ▼
Prediction-Set Size
      │
      ▼
Reliability Weight
      │
      ▼
COME-based Conservative Adaptation
```

Instead of treating all target samples equally during adaptation, AdaPS-COME explicitly controls the influence of samples according to their estimated ambiguity.

---

## Contributions

The main contributions of AdaPS-COME are:

1. **Adaptive prediction-set reliability**

   Prediction-set size is used as a model-induced proxy for target-sample ambiguity.

2. **Uncertainty-adaptive thresholding**

   The prediction-set threshold is dynamically adjusted according to batch-level uncertainty.

3. **Reliability-aware adaptation**

   Samples with larger prediction sets receive smaller adaptation weights, reducing their influence on online model updates.

4. **Combination with conservative TTA**

   The reliability mechanism is integrated with a COME-style conservative objective.

5. **Cross-family evaluation**

   The method is evaluated in combination with multiple TTA families, including Tent, EATA, and SAR.

---

## Experimental Setup

### Dataset

Experiments are conducted on **ImageNet-C**, which evaluates model robustness under common image corruption shifts.

The experiments consider corruption severities from 1 to 5.

### Model

The experiments use:

```text
Model: ViT-Base
Dataset: ImageNet-C
```

### Baselines

The following TTA methods are considered:

- Tent
- EATA
- SAR
- Tent + COME
- EATA + COME
- SAR + COME
- AdaPS-COME

### Default Configuration

| Setting | Value |
|---|---:|
| Batch size | 64 |
| Optimizer | SGD |
| Learning rate | $1\times10^{-3}$ |
| Momentum | 0.9 |
| Weight decay | 0 |
| Updates per batch | 1 |
| $\lambda$ | 0.03 |
| COME p-norm | 2 |
| $q_{\min}$ | 0.90 |
| $q_{\max}$ | 0.98 |
| $\gamma$ | 0.5 |
| Random seed | 0 |

The target stream follows the recorded corruption/severity order without shuffling.

Metrics are evaluated before the update on the same batch.

---

## Results

### Overall Comparison

The following results compare **Tent-COME** and **AdaPS-COME**:

| Method | Accuracy | ECE | Wrong Confidence | Wrong Uncertainty | AUROC | Confidence Separation | Uncertainty Separation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tent-COME | 60.03 | 4.98 | 0.451 | 0.117 | 0.712 | 0.312 | 0.043 |
| AdaPS-COME | 59.38 | 5.02 | 0.446 | 0.119 | 0.715 | 0.315 | 0.045 |

AdaPS-COME achieves slightly lower average accuracy than Tent-COME in this experiment, while showing:

- lower wrong confidence,
- higher wrong uncertainty,
- higher AUROC,
- stronger confidence separation,
- stronger uncertainty separation.

Therefore, the main effect observed in these experiments is related to **reliability and uncertainty behavior**, rather than an overall accuracy improvement.

---

## Cross-Family Ablation

AdaPS-COME is also evaluated with different TTA families.

| TTA Method | Variant | Wrong Conf. | Wrong Unc. | AUROC | Conf. Sep. | Unc. Sep. |
|---|---|---:|---:|---:|---:|---:|
| Tent | Base | 0.451 | 0.116 | 0.712 | 0.312 | 0.043 |
| Tent | +COME | 0.451 | 0.117 | 0.712 | 0.312 | 0.043 |
| Tent | +AdaPS | 0.446 | 0.119 | 0.715 | 0.315 | 0.045 |
| EATA | Base | 0.524 | 0.061 | 0.706 | 0.312 | 0.033 |
| EATA | +COME | 0.523 | 0.062 | 0.706 | 0.312 | 0.033 |
| EATA | +AdaPS | 0.512 | 0.067 | 0.714 | 0.317 | 0.037 |
| SAR | Base | 0.591 | 0.033 | 0.723 | 0.286 | 0.019 |
| SAR | +COME | 0.591 | 0.033 | 0.721 | 0.286 | 0.019 |
| SAR | +AdaPS | 0.572 | 0.039 | 0.735 | 0.298 | 0.023 |

The ablation results show that the prediction-set reliability mechanism changes the uncertainty-related behavior across different TTA families.

---

## Severity-Aware Behavior

The following results are reported for **Tent + AdaPS** across ImageNet-C severity levels.

| Severity | Accuracy | ECE | Wrong Confidence | Wrong Uncertainty | Prediction Set Size | Reliability Weight |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71.67 | 2.33 | 0.505 | 0.093 | 46.9 | 0.541 |
| 2 | 65.58 | 3.68 | 0.473 | 0.106 | 66.3 | 0.465 |
| 3 | 61.38 | 4.53 | 0.452 | 0.116 | 82.4 | 0.417 |
| 4 | 54.31 | 5.99 | 0.419 | 0.130 | 108.3 | 0.346 |
| 5 | 43.95 | 8.60 | 0.381 | 0.149 | 148.9 | 0.263 |

As corruption severity increases:

- Accuracy decreases.
- ECE increases.
- Prediction-set size increases.
- Reliability weights decrease.
- Wrong uncertainty increases.

This behavior is consistent with the intended reliability mechanism: increasingly ambiguous target samples receive less influence during adaptation.

---

## Repository Structure

A recommended repository structure is:

```text
AdaPS-COME/
│
├── README.md
├── .gitignore
│
├── Codes_EATA_SAR/
│   └── ...
│
└── Codes_TENT/
    └── ...
```

The repository contains the implementation for the experiments described in the paper.

Large datasets, model checkpoints, experiment logs, cache files, and Python bytecode files are not included in the repository.

---

## Installation

Create a Python environment and install the required dependencies.

For example:

```bash
git clone https://github.com/shiyue105/AdaPS-COME.git
cd AdaPS-COME
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not provided, please install the dependencies required by the corresponding experiment scripts.

---

## Dataset

AdaPS-COME is evaluated on **ImageNet-C**.

Please download and prepare ImageNet-C according to the official dataset organization.

The dataset itself is not included in this repository.

After downloading the dataset, configure the dataset path in the corresponding experiment scripts.

---

## Running the Experiments

The repository contains implementations for two main experiment groups:

```text
Codes_TENT/
Codes_EATA_SAR/
```

The `Codes_TENT` directory contains the experiments based on Tent.

The `Codes_EATA_SAR` directory contains experiments based on EATA and SAR.

Before running the experiments, configure:

- Dataset path
- Model/checkpoint path
- GPU device
- Batch size
- Learning rate
- TTA configuration

Example:

```bash
cd Codes_TENT
python <experiment_script>.py
```

The exact script names and paths may depend on the final repository organization.

---

## Metrics

The experiments evaluate both predictive performance and uncertainty behavior.

The reported metrics include:

- **Accuracy**
- **Expected Calibration Error (ECE)**
- **Wrong Confidence**
- **Wrong Uncertainty**
- **AUROC**
- **Confidence Separation**
- **Uncertainty Separation**
- **Prediction-Set Size**
- **Reliability Weight**

These metrics are used to evaluate not only classification performance but also how the model's uncertainty changes under distribution shifts.

---

## Limitations

The current experimental results have several limitations.

### 1. Single-Seed Evaluation

The reported aggregate results are based on seed 0.

Multi-seed experiments are required to quantify variance and statistical stability.

### 2. Limited Per-Sample Artifacts

The current experimental artifacts do not contain all raw per-sample logits or probabilities.

Therefore, some analyses such as:

- high-confidence error rate,
- selective-risk curves,
- AURC,
- E-AURC,
- threshold-based abstention analysis

cannot be reliably reconstructed from the current aggregate results.

### 3. Hyperparameter Sensitivity

Further experiments are needed to study the sensitivity of AdaPS-COME to:

- $\lambda$
- $q_{\min}$
- $q_{\max}$
- $\gamma$

### 4. Reliability Weighting Comparisons

Additional comparisons with alternative reliability signals would be useful, including:

- confidence,
- entropy,
- prediction margin,
- uncertainty,
- prediction-set size.

### 5. Thresholding Strategy

The current method uses an adaptive threshold. A comparison between fixed and adaptive thresholds remains an important direction for further evaluation.

---

## Future Experiments

Potential future experiments include:

- Multi-seed evaluation.
- Per-corruption confidence intervals.
- Sensitivity analysis for $\lambda$.
- Sensitivity analysis for $q_{\min}$ and $q_{\max}$.
- Sensitivity analysis for $\gamma$.
- Comparison with entropy-based weighting.
- Comparison with confidence-based weighting.
- Comparison with margin-based weighting.
- Selective prediction and abstention analysis.
- AURC and E-AURC evaluation.
- More detailed per-corruption analysis.

---

## Citation

If you use this code or method in your research, please cite:

```bibtex
@inproceedings{adapscome2026,
  title={AdaPS-COME: Adaptive Prediction-Set Reliability for Safe Test-Time Adaptation of Vision Transformers},
  author={Anonymous},
  year={2026}
}
```

---

## Acknowledgements

This project builds upon previous work on:

- Vision Transformers
- ImageNet-C
- Test-Time Adaptation
- Tent
- EATA
- SAR
- COME
- Conservative and uncertainty-aware model adaptation

We thank the authors of these works for making their research publicly available.

---

## License

Please add an appropriate license to this repository before public release.

If no license is specified, the repository should not be assumed to grant permission to reuse, modify, or redistribute the code.

---

## Contact

For questions regarding the implementation or experiments, please open an issue in this repository.
