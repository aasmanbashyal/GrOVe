#!/usr/bin/env python3
"""
Script to organize experimental results from scattered directories 
into the proper results/ directory structure.
"""

import os
import shutil
import pandas as pd
import glob
from pathlib import Path
import argparse

def organize_basic_attacks(source_dir, dest_dir):
    """Organize basic attack results from experiments/basic/ to results/basic_attacks/"""
    basic_source = Path(source_dir) / "experiments" / "basic"
    basic_dest = Path(dest_dir) / "basic_attacks"
    
    if not basic_source.exists():
        print(f"  Basic experiments directory not found: {basic_source}")
        return
    
    basic_dest.mkdir(parents=True, exist_ok=True)
    
    # Find all model-dataset combinations
    for model_dataset_dir in basic_source.iterdir():
        if model_dataset_dir.is_dir():
            print(f" Organizing basic attacks for: {model_dataset_dir.name}")
            
            # Copy the entire directory
            dest_path = basic_dest / model_dataset_dir.name
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(model_dataset_dir, dest_path)
            
            # Also copy individual CSV files to the main basic_attacks directory
            for csv_file in model_dataset_dir.glob("*.csv"):
                shutil.copy2(csv_file, basic_dest / csv_file.name)

def organize_advanced_attacks(source_dir, dest_dir):
    """Organize advanced attack results from experiments/advanced/ to results/advanced_attacks/"""
    advanced_source = Path(source_dir) / "experiments" / "advanced"
    advanced_dest = Path(dest_dir) / "advanced_attacks"
    
    if not advanced_source.exists():
        print(f"  Advanced experiments directory not found: {advanced_source}")
        return
    
    advanced_dest.mkdir(parents=True, exist_ok=True)
    
    # Find all model-dataset combinations
    for model_dataset_dir in advanced_source.iterdir():
        if model_dataset_dir.is_dir():
            print(f" Organizing advanced attacks for: {model_dataset_dir.name}")
            
            # Copy the entire directory
            dest_path = advanced_dest / model_dataset_dir.name
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(model_dataset_dir, dest_path)
            
            # Copy individual CSV files from attack subdirectories
            for attack_dir in model_dataset_dir.iterdir():
                if attack_dir.is_dir():
                    for csv_file in attack_dir.glob("*.csv"):
                        # Include attack type in filename
                        new_name = f"{model_dataset_dir.name}_{attack_dir.name}_{csv_file.name}"
                        shutil.copy2(csv_file, advanced_dest / new_name)

def organize_csim_verification(source_dir, dest_dir):
    """Organize CSim verification results"""
    csim_source = Path(source_dir) / "experiments" / "csim"
    csim_dest = Path(dest_dir) / "csim_verification"
    
    if not csim_source.exists():
        print(f"  CSim experiments directory not found: {csim_source}")
        return
    
    csim_dest.mkdir(parents=True, exist_ok=True)
    
    # Copy all CSim results
    if csim_source.exists():
        print(f" Organizing CSim verification results")
        
        # Copy the entire csim directory
        if (csim_dest / "csim").exists():
            shutil.rmtree(csim_dest / "csim")
        shutil.copytree(csim_source, csim_dest / "csim")
        
        # Copy CSV files to main directory
        for csv_file in csim_source.rglob("*.csv"):
            rel_path = csv_file.relative_to(csim_source)
            new_name = str(rel_path).replace(os.sep, "_")
            shutil.copy2(csv_file, csim_dest / new_name)

def organize_comprehensive_evaluation(source_dir, dest_dir):
    """Copy comprehensive evaluation results if they exist"""
    comp_source = Path(dest_dir) / "comprehensive_evaluation"
    
    if comp_source.exists():
        print(f" Comprehensive evaluation results already exist at: {comp_source}")
        for dataset_dir in comp_source.iterdir():
            if dataset_dir.is_dir():
                print(f"  - {dataset_dir.name}: {len(list(dataset_dir.glob('*')))} files")

def create_summary_tables(dest_dir):
    """Create summary tables from all collected results"""
    tables_dir = Path(dest_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print(f" Creating summary tables...")
    
    # Basic attacks summary
    basic_csvs = list((Path(dest_dir) / "basic_attacks").glob("*.csv"))
    if basic_csvs:
        basic_dfs = []
        for csv_file in basic_csvs:
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                df['experiment_type'] = 'basic'
                basic_dfs.append(df)
            except Exception as e:
                print(f"  Error reading {csv_file}: {e}")
        
        if basic_dfs:
            basic_summary = pd.concat(basic_dfs, ignore_index=True)
            basic_summary.to_csv(tables_dir / "basic_attacks_summary.csv", index=False)
            print(f" Created basic attacks summary: {len(basic_dfs)} files, {len(basic_summary)} records")
    
    # Advanced attacks summary
    advanced_csvs = list((Path(dest_dir) / "advanced_attacks").glob("*.csv"))
    if advanced_csvs:
        advanced_dfs = []
        for csv_file in advanced_csvs:
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                df['experiment_type'] = 'advanced'
                # Extract attack type from filename
                filename_parts = csv_file.name.split('_')
                if len(filename_parts) >= 3:
                    df['attack_type'] = filename_parts[2]
                advanced_dfs.append(df)
            except Exception as e:
                print(f"  Error reading {csv_file}: {e}")
        
        if advanced_dfs:
            advanced_summary = pd.concat(advanced_dfs, ignore_index=True)
            advanced_summary.to_csv(tables_dir / "advanced_attacks_summary.csv", index=False)
            print(f" Created advanced attacks summary: {len(advanced_dfs)} files, {len(advanced_summary)} records")
    
    # CSim verification summary
    csim_csvs = list((Path(dest_dir) / "csim_verification").glob("*.csv"))
    if csim_csvs:
        csim_dfs = []
        for csv_file in csim_csvs:
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                df['experiment_type'] = 'csim_verification'
                csim_dfs.append(df)
            except Exception as e:
                print(f"  Error reading {csv_file}: {e}")
        
        if csim_dfs:
            csim_summary = pd.concat(csim_dfs, ignore_index=True)
            csim_summary.to_csv(tables_dir / "csim_verification_summary.csv", index=False)
            print(f" Created CSim verification summary: {len(csim_dfs)} files, {len(csim_summary)} records")
    
    # Overall summary
    all_dfs = []
    if 'basic_summary' in locals():
        all_dfs.append(basic_summary)
    if 'advanced_summary' in locals():
        all_dfs.append(advanced_summary)
    if 'csim_summary' in locals():
        all_dfs.append(csim_summary)
    
    if all_dfs:
        overall_summary = pd.concat(all_dfs, ignore_index=True)
        overall_summary.to_csv(tables_dir / "overall_results_summary.csv", index=False)
        print(f" Created overall summary: {len(overall_summary)} total records")

def copy_visualizations(source_dir, dest_dir):
    """Copy visualization files to results without overriding existing ones"""
    viz_source = Path(source_dir) / "visualizations"
    viz_dest = Path(dest_dir) / "visualizations"
    
    if viz_source.exists():
        print(f"  Copying visualizations...")
        viz_dest.mkdir(parents=True, exist_ok=True)
        
        # Copy all visualization files and directories, preserving existing ones
        for item in viz_source.rglob("*"):
            if item.is_file():
                # Calculate relative path from source
                rel_path = item.relative_to(viz_source)
                dest_path = viz_dest / rel_path
                
                # Create parent directories if needed
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not dest_path.exists():
                    shutil.copy2(item, dest_path)
                else:
                    print(f"  Skipping existing file: {rel_path}")
        
        # Count visualizations
        png_files = list(viz_dest.rglob("*.png"))
        print(f" Total visualization files: {len(png_files)}")
    else:
        print(f"  No visualizations directory found at: {viz_source}")


def main():
    parser = argparse.ArgumentParser(description="Organize experimental results")
    parser.add_argument("--source-dir", default=".", help="Source directory (project root)")
    parser.add_argument("--dest-dir", default="results", help="Destination directory")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    
    print(" Organizing experimental results...")
    print(f"Source: {source_dir.absolute()}")
    print(f"Destination: {dest_dir.absolute()}")
    print("-" * 50)
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize different types of results
    organize_basic_attacks(source_dir, dest_dir)
    organize_advanced_attacks(source_dir, dest_dir)
    organize_csim_verification(source_dir, dest_dir)
    organize_comprehensive_evaluation(source_dir, dest_dir)
    copy_visualizations(source_dir, dest_dir)
    
    # Create summary tables
    create_summary_tables(dest_dir)
    
    
    print("-" * 50)
    print(" Results organization completed!")
    print(f" All results are now organized in: {dest_dir.absolute()}")

if __name__ == "__main__":
    main() 