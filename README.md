# Phecoder: Semantic ICD code ranking for efficient Phecode curation.

PheCoder maps clinical phenotypes (Phecodes) to ICD-9/ICD-10 diagnosis codes using transformer-based semantic embeddings. It evaluates multiple embedding models and ensemble methods to find the most relevant diagnosis codes for each phenotype.

**PyTorch with CUDA**

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.



**Installing Phcoder**

 *As a developer* <br>
 Note : python >=3.10 is required
```
git clone https://github.com/03bennej/phecoder.git
pip install pipx
pipx install poetry
pipx install poetry 
```

 *As a user*
 ```
pip install phecoder
 ```
