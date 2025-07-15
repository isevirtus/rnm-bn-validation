# **RNM-BN Validation**

> **Validate an already-calibrated Bayesian Network built with the _Ranked Nodes Method_ (RNM) against expert elicitation data.**

## 📄 Repository Overview

This repository contains:

1. **`src/bn_fitness.py`** – A calibrated Bayesian Network implemented using the Ranked Nodes Method (RNM), modeling Technical Aptitude, Collaborative Aptitude, and Team Aptitude.  

2. **`src/validation.py`** – A command-line interface (CLI) tool to validate the model. It loads input scenarios from CSV files, feeds them into the network, and compares predicted vs expert-provided distributions using **Brier Score**.

3. **`data/`** – Contains the validation inputs (CSV files) with test scenarios provided by domain experts and results, including:
   - Expert judgments
   - Normalized probabilities
   - Mode
   - Heatmaps
   - Brier scores

4. **`results/`** – Automatically generated outputs containing expected vs predicted results, along with Brier Scores.

5. **`variable_definitions/`** – Supplementary material describing:
   - Description for each variable level  (VL - VH)
   - Concrete examples of what each level means in the context of software projects

## 🗂️ Repository Structure

```
rnm-bn-validation/
├── data/ # Expert inputs and probability distributions
│ └── expert_inputs_and_results.xlsx
│ └── TPN_AT_validation.csv
│ └── TPN_AC_validation.csv
│ └── ...
├── results/ # Outputs generated after validation
│ └── AT_validation_results.csv
│ └── AE_validation_results.csv
│ ...
│
├── src/ # Core code
│ └── bn_fitness.py # RNM Bayesian Network definition
│ └── validation.py # CLI validation runner
│
├── variable_definitions/ # PDF explanations for each variable and its levels
│ └── AT_levels_definitions.pdf
```
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

## Quick Start

Run the validator for each target node:

### How Input Files Are Read
The validation.py script receives all its inputs via command-line arguments, allowing users to validate any node in the network without modifying the code. The general structure of the command is:
```bash
python <path_to_script>/validation.py \
       --file <path_to_data>/<validation_input_file>.csv \
       --target <target_variable> \
       --evidence <evidence_var1> <evidence_var2> ...

--file: path to the CSV file containing expert scenarios.

--target: the node in the network you want to validate.

--evidence: the parent nodes of the target, provided as evidence for inference.
```
Each row in the CSV must include values for all evidence variables and the expected probability distribution for the target variable.

### Example
For validating the variable AE (Team Aptitude), whose parents are AT (Technical Aptitude) and AC (Collaborative Aptitude), the command would be:
```bash
python src/validation.py --file data/TPN_AE_validacao.csv \
                         --target AE \
                         --evidence AT AC
This command will:

Read TPN_AE_validacao.csv from the data/ folder.
```
Use each row’s AT and AC values as input evidence. Run inference on the calibrated RNM Bayesian Network. Compare the predicted and expected distributions.
Calculate the Brier Score for each scenario.

The output is printed to the console and saved in the results/ directory with a structured table including evidence, expected distribution, model prediction, and error metrics.

```bash
### AC (parents: PC_VH … PC_VL)
python src/validation.py --file data/TPN_AC_validacao.csv \
                         --target AC \
                         --evidence PC_VH PC_H PC_M PC_L PC_VL

### AT (parents: Domain, Ecosystem, Language)
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
 
