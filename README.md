# Brains Layers

This project provides tools and scripts for analyzing the performance of BERT on various linguistic tasks. Note that only synthetic data is shared in this repository. For other datasets, please consult the appropriate papers.

## Scripts

- `generate_data.py`: Creates synthetic data following the pattern of the experimental setup in Regev et al. (2024).
- `semantics_layers.py`: Tests the layer-wise performance of BERT on a semantic task.
- `syntax_layers.py`: Tests the layer-wise performance of BERT on a syntactic task.
- `aggregated_bert.py`: Aggregates representations from layer 4 and 12 of a language model for multi-label classification.

## Data

Only synthetic data is shared in this repository. For access to other datasets, please refer to the respective publications.

## References

- Bowman, S. R., Angeli, G., Potts, C., Manning, C. D. 2015. A large annotated corpus for learning natural language inference. Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP). 

- Regev, T.I., Casto, C., Hosseini, E.A. et al. 2024. Neural populations in the language network differ in the size of their temporal receptive windows. Nat Hum Behav 8, 1924–1942. https://doi.org/10.1038/s41562-024-01944-2

- Tjong Kim Sang, E.F., De Meulder, F. 2003. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003, 142– 147.
