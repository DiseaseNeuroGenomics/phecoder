# phecoder

Phecoder maps clinical phenotypes (Phecodes) to diagnosis (ICD) codes using pretrained text embedding models. It evaluates multiple embedding models and ensemble methods to find the most relevant diagnosis codes for each phenotype.

<p align="center">
  <img src="https://raw.githubusercontent.com/DiseaseNeuroGenomics/phecoder/main/figures/fig1.png" alt="Phecoder overview" width="600">
</p>

## Install

```bash
pip install phecoder
```

For GPU support, install [PyTorch with CUDA](https://pytorch.org/get-started/locally/) **before** installing phecoder.

For the interactive review widget:

```bash
pip install 'phecoder[review]'
```

## Quick start

```python
import os
from phecoder import Phecoder

os.environ["HF_HOME"] = "./hf-home"

pc = Phecoder(
    phecodes=["Suicidal ideation", "Depression", "Anxiety"],
    output_dir="./results",
    icd_cache_dir="./icd_cache",
)

pc.run()
pc.build_ensemble()

results = pc.load_results("ensemble-zsum")
```

See the [Usage Guide](guide.md) for model selection, custom ICD data, and all configuration options.

## Citation

If you use phecoder in research, please cite:

> **Phecoder: semantic retrieval for auditing and expanding ICD-based phenotypes in EHR biobanks.**
> Jamie J. R. Bennett, Simone Tomasi, Sonali Gupta, VA Million Veteran Program, Georgios Voloudakis, Panos Roussos, David Burstein (2026).
> doi: [10.64898/2026.01.08.26343725](https://www.medrxiv.org/content/10.64898/2026.01.08.26343725v1)

## Support

Open a [GitHub Issue](https://github.com/DiseaseNeuroGenomics/phecoder/issues) or email jamie.bennett@mssm.edu.
