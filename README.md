# SigClean-Biomedical-Signal-Cleaning-Library
SigClean is a comprehensive Python library for cleaning and preprocessing biomedical signals including ECG, EMG, EEG, and other physiological signals. It provides a complete toolkit for signal filtering, artifact removal, noise reduction, and signal quality assessment.

## Features

### Signal Filtering
- **Butterworth filters:** Lowpass, highpass, bandpass, and bandstop filtering  
- **Notch filters:** Remove power line interference (50/60 Hz)  
- **Adaptive filtering:** LMS adaptive noise cancellation  
- **Savitzky–Golay filtering:** Smooth signals while preserving features  
- **Moving average:** Simple signal smoothing  

### Preprocessing & Cleaning
- **Baseline removal:** Linear detrending, polynomial fitting, median filtering  
- **Artifact removal:** Statistical outlier detection and interpolation  
- **Normalization:** Z-score, min–max, robust, and unit vector normalization  
- **Resampling:** Change sampling frequency with interpolation or `scipy` methods  

### Analysis & Quality Assessment
- **Signal segmentation:** Windowed analysis with overlap  
- **Peak detection:** Find R-peaks, spikes, and other signal features  
- **SNR calculation:** Signal-to-noise ratio estimation  
- **Quality metrics:** Comprehensive signal quality assessment  

### Visualization
- **Signal plotting:** Time-domain visualization with customization  
- **Frequency analysis:** Power spectral density and spectrograms  
- **Before/after comparisons:** Side-by-side filtering results  
- **Multi-signal plots:** Compare multiple signals simultaneously  
- **Quality assessment plots:** Comprehensive signal quality visualization  

---

## API Reference

### Filtering Functions (`sigclean.filters`)
- `butterworth_filter()` – Apply Butterworth filters (**lowpass**, **highpass**, **bandpass**, **bandstop**)
- `notch_filter()` – Remove specific frequency components (e.g., **power line interference**)
- `baseline_removal()` – Remove baseline drift and trends
- `artifact_removal()` – Detect and remove signal artifacts
- `adaptive_filter()` – **LMS adaptive filtering** for noise reduction
- `moving_average_filter()` – Simple moving average smoothing
- `savitzky_golay_filter()` – **Savitzky–Golay** smoothing filter

### Utility Functions (`sigclean.utils`)
- `normalize_signal()` – Signal normalization (**z-score**, **min–max**, **robust**)
- `resample_signal()` – Change sampling frequency  
- `segment_signal()` – Split signal into overlapping windows  
- `calculate_snr()` – Signal-to-noise ratio calculation  
- `detect_peaks()` – Peak detection in signals  
- `remove_outliers()` – Outlier detection and removal  
- `calculate_signal_quality()` – Comprehensive quality assessment  

### Plotting Functions (`sigclean.plot`)
- `plot_signal()` – Basic signal visualization  
- `plot_frequency_spectrum()` – Power spectral density plots  
- `plot_before_after()` – Before/after filtering comparison  
- `plot_multiple_signals()` – Multi-signal visualization  
- `plot_spectrogram()` – Time–frequency analysis  
- `plot_signal_quality_assessment()` – Comprehensive quality plots  
- `plot_filter_response()` – Filter frequency response  

---

## Contributing

I welcome contributions!

1. **Fork** the repository  
2. **Create a feature branch**  
   ```bash
   git checkout -b feature/amazing-feature
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
4. **Push to the branch**
   ````bash
   git push origin feature/amazing-feature
5. **Open a Pull Request**

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use **SigClean** in your research or projects, please cite it as:

```bibtex
@software{sigclean2024,
  author    = {Diptiman Mohanta},
  title     = {SigClean: A Python Library for Biomedical Signal Cleaning},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/diptiman-mohanta/sigclean}
}
```
