# MultiHop-RAG

609 news articles and 2,556 questions that need evidence from two to four of them: inference, comparison and temporal questions, and about 12% null queries the corpus cannot answer (abstention is exercised).

- Source: https://github.com/yixuantt/MultiHop-RAG
- Revision: `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`
- License: [ODC-By-1.0](https://opendatacommons.org/licenses/by/1-0/)
- Questions: 300, documents: 609 (stratified subset of 300, seed 42)

## Citation

Yixuan Tang and Yi Yang. 2024. MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries. arXiv:2401.15391.

## Terms and caveats

The dataset is ODC-By (attribution); the articles in it are third-party news. RAGBench downloads them to your machine and does not redistribute them, but check the publishers' terms before you republish the corpus. A subset keeps the whole corpus: the other articles are the distractors.

Pulled with `ragbench datasets pull`; the files were checked against the checksums pinned in RAGBench.
