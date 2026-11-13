# FOVL Word Embeddings Assignment

This workspace contains a reproducible notebook workflow for the FOVL assignment. The notebook trains 300-dimensional monolingual models from scratch using only the specified IndicCorpV2 assignment subset, compares FastText CBOW, FastText Skip-gram, and Word2Vec Skip-gram with a character-overlap OOV strategy, and writes the submission artifacts.

## What is covered

The public benchmark has Gujarati, Hindi, Marathi, Punjabi, Tamil, Telugu, and Urdu word-pair files. The specified IndicCorpV2 subset provides 5,000 training lines for each of those seven languages. Some benchmark rows do not have a human score: 149 Marathi rows and one Tamil row in the current repository snapshot. Those pairs are still kept in the output vector arrays in their original order; only rows with a human score can be included in Spearman evaluation.

All human scores are used only for model comparison and evaluation, not for training. No pre-trained embeddings or outside training corpora are used.

## Run the notebook

Use Python 3.10+ and install the notebook dependencies in the active environment:

```text
python -m pip install "numpy<2.5" "scipy>=1.13" "gensim>=4.4" datasets pandas matplotlib scikit-learn reportlab
```

Open `FOVL_Word_Embeddings.ipynb` and run the cells from top to bottom. The notebook downloads IndicCorpV2 through Hugging Face Datasets and caches the public word-similarity files from the repository named in the assignment. Internet access is needed for the initial run. Training and evaluation can take a while; start with the smoke-test cell before running both model candidates.

The notebook selects one of its three tested candidates globally by the macro-average Spearman correlation over the seven languages. It trains one monolingual model per language. FastText subword vectors handle OOV words for the FastText candidates; the Word2Vec candidate uses a weighted average of the five vocabulary vectors with the greatest character n-gram overlap. Multi-token expressions are represented by averaging the vectors of their component tokens. The selected candidate is the best among those tested on the public benchmark, not a proof that no other approach or setting could do better.

## Generated files

The notebook writes its results under `outputs\`:

- `pair_embeddings\`: one `.npy` array per language, with shape `(N, 2, 300)` and the source pair order preserved.
- `vocabularies\`: a JSON token-to-row mapping and its corresponding `.npy` vector matrix for each language.
- `models\`: selected FastText keyed-vector files, including subword information needed for OOV inference.
- `results.csv`, `analysis_examples.csv`, and `tsne\`: evaluation tables, examples, and one 100-word visualization per language.
- `FOVL_Experimental_Report.pdf`: the assignment report.
- `FOVL_Submission.zip`: the notebook, README, report, and required output artifacts.

The model files are included so the notebook can reload a trained model and produce vectors for new words without retraining. The notebook also demonstrates that inference path.

## Limitations

The assignment dataset is a 5,000-line-per-language subset, so models trained on it are small and should not be treated as production-quality embeddings. The Marathi and Tamil public files contain unscored pairs; those cannot contribute to correlation statistics but remain part of the vector submission. Results depend on the exact public dataset snapshots downloaded when the notebook is run.

## References

- Doddapaneni et al. (2023), IndicCorp v2, [ACL Anthology](https://aclanthology.org/2023.acl-long.693/).
- Akhtar et al. (2017), Word Similarity Datasets for Indian Languages, [public repository](https://github.com/syedsarfarazakhtar/Word-Similarity-Datasets-for-Indian-Languages).
- [IndicCorpV2-5k assignment subset](https://huggingface.co/datasets/rony000013/indiccorpV2-5k).
