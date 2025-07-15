# **RNM-BN Validation**

> **Validate an already-calibrated Bayesian Network built with the _Ranked Nodes Method_ (RNM) against expert elicitation data.**

This repository contains:

1. **`bn_fitness.py`** – a calibrated RNM Bayesian Network that models  
  *AT* (Technical Aptitude), *AC* (Collaborative Aptitude) and *AE* (Team Aptitude).
2. **`validation.py`** – a CLI tool that reads CSV scenarios provided by specialists, feeds them as evidence into the network, and reports **expected vs predicted** distributions together with the **Brier Score** for every case.

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
      ...
├── results/                    # Auto-generated validation outputs
├── src/
│   ├── bn_fitness.py           # Calibrated RNM Bayesian Network
│   └── validation.py           # Command-line validator
└── repository.json             # Continuous samples used by RNM functions
```

---

## Quick Start

Run the validator for each target node:

# How Input Files Are Read
The validation.py script receives all its inputs via command-line arguments, allowing users to validate any node in the network without modifying the code. The general structure of the command is:
```bash
python <path_to_script>/validation.py \
       --file <path_to_data>/<validation_input_file>.csv \
       --target <target_variable> \
       --evidence <evidence_var1> <evidence_var2> ...

--file: path to the CSV file containing expert scenarios.

--target: the node in the network you want to validate.
```
--evidence: the parent nodes of the target, provided as evidence for inference.

Each row in the CSV must include values for all evidence variables and the expected probability distribution for the target variable.

# Example
For validating the variable AE (Team Aptitude), whose parents are AT (Technical Aptitude) and AC (Collaborative Aptitude), the command would be:

python src/validation.py --file data/TPN_AE_validacao.csv \
                         --target AE \
                         --evidence AT AC
This command will:

Read TPN_AE_validacao.csv from the data/ folder.

Use each row’s AT and AC values as input evidence. Run inference on the calibrated RNM Bayesian Network. Compare the predicted and expected distributions.
Calculate the Brier Score for each scenario.

The output is printed to the console and saved in the results/ directory with a structured table including evidence, expected distribution, model prediction, and error metrics.

```bash
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

| Var1 | Var n | Expected_VL | Expected_L | Expected_M | Expected_H | Expected_VH | Calculated_VL | … | Brier_Score |
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
 
