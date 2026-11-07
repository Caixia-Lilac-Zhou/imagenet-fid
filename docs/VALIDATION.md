# Local Release Validation

Validated on 2026-11-06 with Python 3.12.14, NumPy 2.5.4, Matplotlib 3.11.2, and Pillow 13.0.0. The published dependency ranges are compatibility bounds, not a complete lockfile for the historical GPU environment.

| Check | Result |
| --- | --- |
| Original FD moment calculations and identity distance | Passed on NumPy |
| Stratified bootstrap determinism and interrupted-run recovery | Passed |
| Generated/reference subsampling determinism and recovery | Passed |
| Fresh-run collision and mismatched-resume rejection | Passed; prior files preserved |
| Portable manifest and full-pool point evaluation | Passed |
| Paper value regression tests | Passed |
| Small end-to-end synthetic CPU example | Passed |
| Four main figures rebuilt from released CSVs | Passed; PNG pixels match camera-ready previews |
| Main PDF figure format | Four single-page vector PDFs, no embedded raster images |
| Released data checksums and Python syntax | Passed |
| Original-model source registry coverage | Passed on 2026-11-06: all 47 labels and 15 families; MeanFlow maps to AlphaFlow; pMF uses the author-confirmed PyTorch `torch` branch; legacy LightningDiT keys retained |
| Private absolute-path, IP-address, and token-pattern scan | No findings in the public allowlist |
| Automatic file/directory deletion call scan | No findings in released Python code |
| PyTorch numerical comparison and synthetic extractor tests | Not rerun locally: PyTorch unavailable |
| Real GPU feature extraction or new full-size image generation | Not rerun as part of repository preparation |

Tests use tiny synthetic features and retain their temporary files. This validation does not replace independent GPU reproduction or the missing historical provenance documented in `PROVENANCE.md`. Pattern scans are release checks, not a guarantee against every possible secret or unsafe third-party behavior.
