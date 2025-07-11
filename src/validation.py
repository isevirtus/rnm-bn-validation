import os
import numpy as np
import pandas as pd
from bn_fitness import FitnessBayesianNetwork,STATES




def calculate_brier_score(expected, predicted):
    expected = np.array([float(x.replace(',', '.')) for x in expected], dtype=float)
    predicted = np.array(predicted, dtype=float)
    return np.mean((expected - predicted) ** 2)

  

if __name__ == "__main__":
    net = FitnessBayesianNetwork()
    
    
    import argparse

    parser = argparse.ArgumentParser(description="Validate Bayesian Network distributions.")
    parser.add_argument("--file", required=True, help="CSV file with validation data.")
    parser.add_argument("--target", required=True, choices=["AT", "PC", "AE", "AC"], help="Node to validate.")
    parser.add_argument("--evidence", nargs="+", required=True, help="Evidence columns (e.g. Dom Eco Ling).")
    args = parser.parse_args()

    validation_file = args.file
    target_node = args.target
    evidence_cols = args.evidence

    validation_data = pd.read_csv(validation_file)

    
    
    results = []
    
    print("="*60)
    print("VALIDATING AT DISTRIBUTIONS AGAINST TEST DATA")
    print("="*60)
    
    total_brier = 0
    num_cases = 0
    
    for idx, row in validation_data.iterrows():
        # extract evidence
        evidence = {col: row[col] for col in evidence_cols}
        expected_dist = [row['VL'], row['L'], row['M'], row['H'], row['VH']]

        
        # Ccalculate distribution
        net.evidence = {}
        for node, state in evidence.items():
            net.set_evidence(node, state)
        
        calculated_dist = net.predict(target_node)

        
        # metric 
        brier = calculate_brier_score(expected_dist, calculated_dist)
        total_brier += brier
        num_cases += 1
        
        # results
        results.append({
            'Evidence': evidence,
            'Expected_VL': expected_dist[0],
            'Expected_L': expected_dist[1],
            'Expected_M': expected_dist[2],
            'Expected_H': expected_dist[3],
            'Expected_VH': expected_dist[4],
            'Calculated_VL': calculated_dist[0],
            'Calculated_L': calculated_dist[1],
            'Calculated_M': calculated_dist[2],
            'Calculated_H': calculated_dist[3],
            'Calculated_VH': calculated_dist[4],
            'Brier_Score': brier
        })
        
        print(f"\n🔍 Case {idx+1}: Evidence = {evidence}")
        print("📊 Expected vs Calculated:")
        print(f"  {'State':<5} | {'Expected':<8} | {'Calculated':<8}")
        for state, exp, calc in zip(STATES, expected_dist, calculated_dist):
            exp = float(exp.replace(",", "."))
            calc = float(calc)
            print(f"  {state:<5} | {exp:<8.4f} | {calc:<8.4f}")
        print(f"📌 Brier Score: {brier:.6f}")
    
    # Brier 
    mean_brier = total_brier / num_cases if num_cases > 0 else 0
    
    # save CSV
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f'{target_node}_test_results.csv')

    
    
    df_results = pd.DataFrame(results)

    
    evidence_keys = evidence_cols  
    df_export = pd.DataFrame()

    
    for key in evidence_keys:
        df_export[key] = [str(e[key]) for e in df_results['Evidence']]

    
    for key in evidence_keys:
        df_export[key] = [str(e[key]) for e in df_results['Evidence']]

    # add colum Expected : VL, L, M, H, VH
    expected_cols = [f'Expected_{s}' for s in STATES]
    calculated_cols = [f'Calculated_{s}' for s in STATES]

    for col in expected_cols + calculated_cols:
        df_export[col] = df_results[col]



    # Score
    df_export['Brier_Score'] = df_results['Brier_Score']

    
    df_export.to_csv(output_file, index=False, float_format='%.4f')
    
    print("\n" + "="*60)
    print(f"✅ VALIDATION for {target_node} COMPLETED - Mean Brier Score: {mean_brier:.6f}")

    print(f"Results saved to: {output_file}")
    print("="*60)