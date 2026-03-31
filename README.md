# Mach Number & Structure Function Analysis

This repository provides a Python-based implementation for diagnosing interstellar medium (ISM) turbulence. The project focuses on estimating **Sonic ($M_s$)** and **Alfvénic ($M_a$) Mach numbers** and constructing **Structure Functions (SF)** based on the theoretical framework developed by **A. Lazarian** and collaborators (2025).

## Project Structure

To maintain a clean environment and ensure the pipeline functions correctly, please adhere to the following directory layout:

```text
.
├── data/
│   ├── BISTRO_Clean/       # Processed BISTRO/JCMT data
│   └── SALSA_Clean/        # Processed SALSA data
├── clean_dir.py            # Utility script for data standardization
├── pipeline_code_bistro.ipynb
├── ...
├── .gitignore              # Configured to ignore /data and /report
└── [report]/                # Generated PDF visualizations (created automatically)
```

## Workflow
### 1. Data Preparation
Raw data sorted by objects should be processed using clean_dir.py. This script standardizes the directory structure and moves the cleaned files into the data/ subdirectory.

Note: The data/ directory is ignored by Git to avoid repository bloating.

### 2. Analysis Pipeline
The core analysis is performed within pipeline_code_bistro.ipynb.

Origin: This pipeline is an evolution of the bistro_JCMT.ipynb framework.

Operation: It processes data directly from the data/BISTRO_Clean folder but you can change it.

Output: All resulting plots are saved as high-quality PDFs in the root directory.

## Theoretical Foundation
The implementation leverages recent advances in MHD turbulence theory, specifically focusing on the anisotropy of gradient directions and polarization structure functions to derive magnetization in super-Alfvénic regimes.

## References
[1] Lazarian, A., Hu, Y., and Pogosyan, D., “Obtaining Magnetization of Super-Alfvénic Turbulence with the Structure Functions of Gradient Directions”, arXiv e-prints, Art. no. arXiv:2512.19816, 2025. doi:10.48550/arXiv.2512.19816.

[2] Lazarian, A., Pogosyan, D., and Hu, Y., “Model of super-Alfvénic MHD turbulence and structure functions of polarization”, arXiv e-prints, Art. no. arXiv:2511.08800, 2025. doi:10.48550/arXiv.2511.08800.
