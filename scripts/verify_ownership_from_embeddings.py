#!/usr/bin/env python3
"""
Verify GNN model ownership using trained Csim and pre-saved embeddings.

This script implements the ownership verification as specified in the task:
- Uses pre-trained Csim similarity model
- Compares embeddings from target and suspect models
- Generates element-wise squared distance vectors
- Classifies similarity to determine if suspect is surrogate or independent
"""

import argparse
import torch
import os
import sys
import numpy as np
from pathlib import Path

# Add grove to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grove.verification.similarity_model import  CsimManager


def main():
    parser = argparse.ArgumentParser(description='Verify model ownership using saved embeddings')
    parser.add_argument('--target-model-name', type=str, required=True,
                       help='Target model name (e.g., gat_citeseer_target)')
    parser.add_argument('--target-embedding', type=str, required=True,
                       help='Path to target model embeddings')
    parser.add_argument('--suspect-embedding', type=str, required=True,
                       help='Path to suspect model embeddings')
    parser.add_argument('--csim-model-dir', type=str, default='models/csim',
                       help='Directory containing trained Csim models')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold for classification (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CSIM OWNERSHIP VERIFICATION")
    print("="*60)
    print(f"Target model: {args.target_model_name}")
    print(f"Target embedding: {args.target_embedding}")
    print(f"Suspect embedding: {args.suspect_embedding}")
    print(f"Csim model directory: {args.csim_model_dir}")
    print(f"Threshold: {args.threshold}")
    print("="*60)
    
    try:
        # Verify files exist
        if not os.path.exists(args.target_embedding):
            raise FileNotFoundError(f"Target embedding not found: {args.target_embedding}")
        
        if not os.path.exists(args.suspect_embedding):
            raise FileNotFoundError(f"Suspect embedding not found: {args.suspect_embedding}")
        
        # Create Csim manager
        manager = CsimManager(base_save_dir=args.csim_model_dir)
        
        # Verify ownership
        print(f"\n🔍 Verifying ownership...")
        results = manager.verify_from_embeddings(
            target_model_name=args.target_model_name,
            target_embedding_path=args.target_embedding,
            suspect_embedding_path=args.suspect_embedding,
            threshold=args.threshold
        )
        # Append results to a CSV file
        import csv
        from pathlib import Path
        import datetime

        # Define CSV file path (in csim_model_dir)
        csv_file = Path(args.csim_model_dir) / "verification_results.csv"

        # Prepare row data
        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "target_model": args.target_model_name,
            "target_embedding": args.target_embedding,
            "suspect_embedding": args.suspect_embedding,
            "total_pairs": results.get("total_pairs", ""),
            "num_similar": results.get("num_similar", ""),
            "similarity_percentage": results.get("similarity_percentage", ""),
            "mean_similarity_prob": results.get("mean_similarity_prob", ""),
            "threshold": results.get("threshold", ""),
            "decision": results.get("decision", ""),
            "is_surrogate": results.get("is_surrogate", ""),
        }

        # Write header if file does not exist
        write_header = not csv_file.exists()
        try:
            with open(csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(f"\n📝 Results appended to: {csv_file}")
        except Exception as e:
            print(f"⚠️  Could not write results to CSV: {e}")
        # Display results
        print(f"\n" + "="*60)
        print(f"VERIFICATION RESULTS")
        print(f"="*60)
        print(f"Target Model: {args.target_model_name}")
        print(f"Suspect Model: {Path(args.suspect_embedding).name}")
        print(f"")
        print(f"📊 Similarity Analysis:")
        print(f"   Total embedding pairs: {results['total_pairs']}")
        print(f"   Similar pairs: {results['num_similar']}")
        print(f"   Similarity percentage: {results['similarity_percentage']:.1%}")
        print(f"   Mean similarity probability: {results['mean_similarity_prob']:.4f}")
        print(f"   Threshold used: {results['threshold']}")
        print(f"")
        print(f"🎯 Final Decision: {results['decision'].upper()}")
        if results['is_surrogate']:
            print(f"   ⚠️  The suspect model is likely a SURROGATE of the target model")
            print(f"   🔴 Potential intellectual property theft detected!")
        else:
            print(f"   ✅ The suspect model appears to be INDEPENDENT")
            print(f"   🟢 No evidence of model stealing detected")
        print(f"="*60)
        
        return results['is_surrogate']
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    is_surrogate = main()
    sys.exit(1 if is_surrogate else 0) 