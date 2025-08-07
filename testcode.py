#!/usr/bin/env python3
"""
Example usage of SigClean library for biomedical signal processing.
This script reads data from the 3rd column of a CSV file and processes it.
Run this script to test the library functionality with real data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import argparse

# Add the sigclean package to the path (if running from the project directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import sigclean functions
from sigclean.filters import butterworth_filter, notch_filter, baseline_removal, artifact_removal
from sigclean.utils import calculate_snr, normalize_signal
from sigclean.plot import plot_before_after, plot_frequency_spectrum

def load_csv_data(csv_file, column_index=2, sampling_frequency=1000):
    """
    Load data from CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    column_index : int
        Column index to read (0-based, default=2 for 3rd column)
    sampling_frequency : float
        Sampling frequency of the data in Hz
    
    Returns:
    --------
    data : ndarray
        Signal data from the specified column
    fs : float
        Sampling frequency
    """
    try:
        # Read CSV file
        print(f"Loading data from: {csv_file}")
        df = pd.read_csv(csv_file)
        
        print(f"CSV shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        
        # Check if the requested column exists
        if column_index >= df.shape[1]:
            raise ValueError(f"Column index {column_index} doesn't exist. CSV has {df.shape[1]} columns (0-indexed).")
        
        # Extract the 3rd column (index 2)
        column_name = df.columns[column_index] if column_index < len(df.columns) else f"Column_{column_index}"
        data = df.iloc[:, column_index].values
        
        print(f"Using column: '{column_name}' (index {column_index})")
        print(f"Data points: {len(data)}")
        print(f"Data range: {np.min(data):.4f} to {np.max(data):.4f}")
        
        # Remove any NaN values
        if np.any(np.isnan(data)):
            print(f"Warning: Found {np.sum(np.isnan(data))} NaN values. Removing them.")
            data = data[~np.isnan(data)]
        
        # Remove any infinite values
        if np.any(np.isinf(data)):
            print(f"Warning: Found {np.sum(np.isinf(data))} infinite values. Removing them.")
            data = data[~np.isinf(data)]
        
        print(f"Final data length: {len(data)} samples")
        print(f"Duration: {len(data)/sampling_frequency:.2f} seconds at {sampling_frequency} Hz")
        
        return data, sampling_frequency, column_name
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")

def create_sample_csv(filename="sample_ecg_data.csv", duration=10, fs=1000):
    """
    Create a sample CSV file with synthetic ECG data for testing.
    
    Parameters:
    -----------
    filename : str
        Output CSV filename
    duration : float
        Duration in seconds
    fs : float
        Sampling frequency
    """
    print(f"Creating sample CSV file: {filename}")
    
    t = np.linspace(0, duration, int(fs * duration))
    
    # Generate synthetic data for multiple columns
    np.random.seed(42)
    
    # Column 1: Time stamps
    timestamps = t
    
    # Column 2: Some other signal
    other_signal = np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(len(t))
    
    # Column 3: ECG-like signal with noise (this is what we'll process)
    ecg_clean = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 3.6 * t)
    noise = 0.2 * np.random.randn(len(t))
    powerline = 0.15 * np.sin(2 * np.pi * 50 * t)
    baseline_drift = 0.1 * t * 0.1 + 0.05 * np.sin(2 * np.pi * 0.1 * t)
    
    ecg_noisy = ecg_clean + noise + powerline + baseline_drift
    
    # Add some artifacts
    artifact_indices = np.random.choice(len(t), size=20, replace=False)
    ecg_noisy[artifact_indices] += np.random.normal(0, 2, len(artifact_indices))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time': timestamps,
        'Other_Signal': other_signal,
        'ECG_Signal': ecg_noisy  # This will be column index 2 (3rd column)
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Sample CSV created with {len(df)} rows and {len(df.columns)} columns")
    return filename

def main():
    """Main function demonstrating SigClean usage with CSV data."""
    
    parser = argparse.ArgumentParser(description='Process biomedical signals from CSV file using SigClean')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--column', type=int, default=2, help='Column index to process (0-based, default=2)')
    parser.add_argument('--fs', type=float, default=1000, help='Sampling frequency in Hz (default=1000)')
    parser.add_argument('--create-sample', action='store_true', help='Create a sample CSV file for testing')
    
    args = parser.parse_args()
    
    print("=== SigClean Library Demo with CSV Data ===\n")
    
    # Create sample CSV if requested
    if args.create_sample:
        sample_file = create_sample_csv()
        print(f"Sample CSV created: {sample_file}")
        if not args.csv:
            args.csv = sample_file
        print()
    
    # Determine CSV file to use
    csv_file = args.csv
    if not csv_file:
        # Look for common CSV files in current directory
        possible_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if possible_files:
            csv_file = possible_files[0]
            print(f"No CSV specified, using: {csv_file}")
        else:
            print("No CSV file specified and none found in current directory.")
            print("Creating a sample file for demonstration...")
            csv_file = create_sample_csv()
            print()
    
    try:
        # Load data from CSV
        print("1. Loading data from CSV file...")
        signal_data, fs, column_name = load_csv_data(csv_file, args.column, args.fs)
        
        if len(signal_data) == 0:
            raise ValueError("No valid data found in the specified column.")
        
        print(f"   - Successfully loaded {len(signal_data)} samples")
        print(f"   - Column: {column_name}")
        print(f"   - Duration: {len(signal_data)/fs:.2f} seconds\n")
        
        # Basic signal statistics
        print("2. Initial signal analysis...")
        print(f"   - Mean: {np.mean(signal_data):.4f}")
        print(f"   - Std: {np.std(signal_data):.4f}")
        print(f"   - Min: {np.min(signal_data):.4f}")
        print(f"   - Max: {np.max(signal_data):.4f}")
        
        # Estimate initial SNR (using high-frequency noise as reference)
        try:
            initial_snr = calculate_snr(signal_data)
            print(f"   - Estimated SNR: {initial_snr:.2f} dB\n")
        except:
            print("   - Could not estimate initial SNR\n")
        
        # Step-by-step signal cleaning
        print("3. Cleaning the signal...")
        
        # Step 1: Remove powerline interference (50 Hz and 60 Hz)
        print("   - Removing powerline interference (50 Hz)...")
        signal_step1 = notch_filter(signal_data, fs, freq=50.0, quality_factor=30)
        
        print("   - Removing powerline interference (60 Hz)...")
        signal_step1 = notch_filter(signal_step1, fs, freq=60.0, quality_factor=30)
        
        # Step 2: Apply bandpass filter (adjust based on signal type)
        print("   - Applying bandpass filter...")
        # For ECG: 0.5-40 Hz, for EMG: 20-450 Hz, for EEG: 0.5-45 Hz
        # Using ECG range as default
        signal_step2 = butterworth_filter(signal_step1, fs, 'bandpass', 
                                         low_freq=0.5, high_freq=min(40, fs/3), order=4)
        
        # Step 3: Remove baseline drift
        print("   - Removing baseline drift...")
        signal_step3 = baseline_removal(signal_step2, method='detrend')
        
        # Step 4: Remove artifacts
        print("   - Detecting and removing artifacts...")
        signal_step4, artifact_mask = artifact_removal(signal_step3, fs, threshold_std=3.0)
        
        # Step 5: Normalize the signal
        print("   - Normalizing signal...")
        signal_final, norm_params = normalize_signal(signal_step4, method='zscore')
        
        print(f"   - Artifacts detected: {np.sum(artifact_mask)} samples ({np.sum(artifact_mask)/len(signal_data)*100:.1f}%)")
        
        # Calculate final SNR
        try:
            final_snr = calculate_snr(signal_final)
            improvement = final_snr - initial_snr
            print(f"   - Final SNR: {final_snr:.2f} dB")
            print(f"   - SNR improvement: {improvement:.2f} dB\n")
        except:
            print("   - Could not calculate final SNR\n")
        
        # Visualization
        print("4. Creating visualizations...")
        
        # Plot before and after comparison
        fig1, axes1 = plot_before_after(signal_data, signal_final, fs,
                                        titles=(f'Original Signal ({column_name})', 'Cleaned Signal'),
                                        colors=('red', 'blue'))
        plt.suptitle(f'SigClean: Signal Processing Results\nFile: {os.path.basename(csv_file)}', 
                     fontsize=14, fontweight='bold')
        
        # Plot frequency spectra
        fig2, ax2, freqs2, psd2 = plot_frequency_spectrum(
            signal_data, fs, title="Original Signal - Frequency Spectrum",
            freq_range=(0, min(100, fs/2))
        )
        
        fig3, ax3, freqs3, psd3 = plot_frequency_spectrum(
            signal_final, fs, title="Cleaned Signal - Frequency Spectrum", 
            freq_range=(0, min(100, fs/2))
        )
        
        # Show artifact locations if any were detected
        if np.any(artifact_mask):
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            time_axis = np.arange(len(signal_data)) / fs
            
            # Show only a portion of the signal if it's too long
            if len(signal_data) > 10000:  # If more than 10k samples, show first 10 seconds
                end_idx = min(int(10 * fs), len(signal_data))
                time_plot = time_axis[:end_idx]
                signal_plot = signal_data[:end_idx]
                artifacts_plot = artifact_mask[:end_idx]
                title_suffix = f" (First {end_idx/fs:.1f} seconds)"
            else:
                time_plot = time_axis
                signal_plot = signal_data
                artifacts_plot = artifact_mask
                title_suffix = ""
            
            ax4.plot(time_plot, signal_plot, 'b-', alpha=0.7, label='Original Signal', linewidth=0.8)
            if np.any(artifacts_plot):
                ax4.plot(time_plot[artifacts_plot], signal_plot[artifacts_plot], 'ro', 
                        markersize=4, label=f'Detected Artifacts ({np.sum(artifact_mask)} total)')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Amplitude')
            ax4.set_title(f'Artifact Detection Results{title_suffix}')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        print("   - Plots created successfully!")
        print("\n5. Displaying results...")
        
        # Show all plots
        plt.tight_layout()
        plt.show()
        
        print("\n=== Processing completed successfully! ===")
        print(f"\nThe SigClean library successfully processed '{column_name}' from {os.path.basename(csv_file)}:")
        print("✓ Removed powerline interference")
        print("✓ Applied frequency filtering")
        print("✓ Removed baseline drift")
        print("✓ Detected and interpolated artifacts")
        print("✓ Normalized the signal")
        if 'improvement' in locals():
            print(f"✓ Improved SNR by {improvement:.2f} dB")
        
        # Save processed data option
        save_output = input("\nDo you want to save the processed signal to a CSV file? (y/n): ").lower().strip()
        if save_output in ['y', 'yes']:
            output_file = f"cleaned_{os.path.basename(csv_file)}"
            output_df = pd.DataFrame({
                'Time': np.arange(len(signal_final)) / fs,
                'Original_Signal': signal_data,
                'Cleaned_Signal': signal_final,
                'Artifacts_Detected': artifact_mask.astype(int)
            })
            output_df.to_csv(output_file, index=False)
            print(f"Processed data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("pip install numpy scipy matplotlib pandas")
        print("\nAnd that you're running this script from the project directory.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()