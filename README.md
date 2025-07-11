# **RNM-BN Validation Suite**

> **Validate an already-calibrated Bayesian Network built with the _Ranked Nodes Method_ (RNM) against expert elicitation data.**

This repository contains:

1. **`bn_fitness.py`** – a calibrated RNM Bayesian Network that models  
   *AT* (Aptidão Técnica), *PC* (Peso de Colaboração), *AC* (Aptidão Colaborativa) e *AE* (Aptidão de Equipe).
2. **`validation.py`** – a CLI tool that reads CSV scenarios provided by specialists, feeds them as evidence into the network, and reports **expected vs predicted** distributions together with the **Brier Score** for every case.

> **Terminologia**  
> • **OSF** – *Objective Success Factor* (fator de sucesso objetivo)  
> • **SLF** – *Subjective Leadership Factor* (fator de liderança subjetiva)

---

## 📦 Installation

```bash
git clone https://github.com/<your-user>/rnm-bn-validation.git
cd rnm-bn-validation
pip install numpy pandas matplotlib pgmpy
```

> No genetic algorithm, no calibration step – the network is already tuned.  
> You only need the Python dependencies above.

---

## 🗂️ Repository Structure

```
rnm-bn-validation/
├── data/                       # CSV test suites from domain experts
│   ├── TPN_AT_validacao.csv
│   ├── TPN_PC_validacao.csv
│   ├── TPN_AC_validacao.csv
│   └── TPN_AE_validacao.csv
├── results/                    # Auto-generated validation outputs
├── src/
│   ├── bn_fitness.py           # Calibrated RNM Bayesian Network
│   └── validation.py           # Command-line validator
└── repository.json             # Continuous samples used by RNM functions
```

---

## 🚀 Quick Start

Run the validator for each target node:

```bash
# PC  (parents: OSF, SLF)
python src/validation.py --file data/TPN_PC_validacao.csv \
                         --target PC \
                         --evidence OSF SLF

# AE (parents: AT, AC)
python src/validation.py --file data/TPN_AE_validacao.csv \
                         --target AE \
                         --evidence AT AC

# AC (parents: PC_VH … PC_VL)
python src/validation.py --file data/TPN_AC_validacao.csv \
                         --target AC \
                         --evidence PC_VH PC_H PC_M PC_L PC_VL

# AT (parents: Domain, Ecosystem, Language)
python src/validation.py --file data/TPN_AT_validacao.csv \
                         --target AT \
                         --evidence Domain Ecosystem Language
```

---

## ✅ What you get

Console output showing, for every scenario:

- Evidence supplied  
- Expected distribution (expert)  
- Predicted distribution (model)  
- Brier Score

A tidy CSV is saved under `results/`, e.g. `results/PC_validation_results.csv`, containing:

| OSF | SLF | Expected_VL | Expected_L | Expected_M | Expected_H | Expected_VH | Calculated_VL | … | Brier_Score |
|-----|-----|--------------|------------|------------|------------|--------------|----------------|---|--------------|

---

## 🖥️ Working on the Network

If you wish to inspect or extend the RNM network:

```python
from src.bn_fitness import FitnessBayesianNetwork
net = FitnessBayesianNetwork()
net.visualize_network()
```

The `repository.json` file provides ≥10 000 continuous samples for each linguistic state (`VL`, `L`, `M`, `H`, `VH`) required by the truncated-normal transformation used internally.
