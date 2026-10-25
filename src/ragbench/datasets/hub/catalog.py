"""The built-in hub datasets. Every file is pinned to a commit of its repository and to a checksum from that repository's own file listing
(`sha256` for LFS files, the git blob id for the small JSON ones), so a pull gets exactly these bytes or fails.

Re-pin from the listing (`GET https://huggingface.co/api/datasets/<repo>?blobs=true`) when moving to a newer revision, and check the license again.
CRAG (CC-BY-NC-4.0, multi-gigabyte web pages) and FRAMES (its corpus is live Wikipedia, which cannot be pinned) are deliberately not here.
"""

from __future__ import annotations

from ragbench.datasets.hub.registry import HubDataset, HubFile, register

GALILEO_REVISION = "97808f3e5fd16ede40bbff6c2949af8139b2eb7b"
# subset -> (path in the repository, size in bytes, sha256), from the repository's file listing at GALILEO_REVISION
GALILEO_TEST_FILES = {
    "covidqa": ("covidqa/test-00000-of-00001.parquet", 854068, "25bc50f064a0b1de82d97eef2b7d7a71b5e2364522f3622ba947a61763f94174"),
    "cuad": ("cuad/test-00000-of-00001.parquet", 12776734, "37c52fd2b9282b572289c2d42f66fd2fce2b7cf187fe8b3c22f2e90c4b866545"),
    "delucionqa": ("delucionqa/test-00000-of-00001.parquet", 561783, "6d20b87a323a5b1435b8b37ceb18ff90a0eb5b99358bf0c80397b222ac902bda"),
    "emanual": ("emanual/test-00000-of-00001.parquet", 304626, "ba6164b48ec0792f9072f952af60062a14d1412dd472ddf08bd50b7006fc4504"),
    "expertqa": ("expertqa/test-00000-of-00001.parquet", 2806982, "30bb2eee7d013356e247ec1a3130b35461c68c843b3660984defafe96ca4de42"),
    "finqa": ("finqa/test-00000-of-00001.parquet", 8936570, "b55bd373741fbe65933191690267421183f898c8253f5db2a4a4703360ecf94d"),
    "hagrid": ("hagrid/test-00000-of-00001.parquet", 3965263, "a047dc86a2cdb16ddbf7f7f20e8b8219c7f0c3951ec947ca63cbb9388cece349"),
    "hotpotqa": ("hotpotqa/test-00000-of-00001.parquet", 1314681, "e8fe09d4a8a7979f9417c5c248991e8f162012f4b1e54177b96de14b35eaa502"),
    "msmarco": ("msmarco/test-00000-of-00001.parquet", 2119223, "cb0d7d2cc4191df8d960512a127fec6675f1aadc7da832e504ac5982a94a2b7f"),
    "pubmedqa": ("pubmedqa/test-00000-of-00001.parquet", 10269730, "8b59334cef14adbc36e199179512dc7803d0560c0a707ce4ae0e094cf683c38d"),
    "tatqa": ("tatqa/test-00000-of-00001.parquet", 4746214, "136d245b28fc6381abe00946ca1f7cab5f3d64ba1847e49e54a2585ec0ec4dab"),
    "techqa": ("techqa/test-00000-of-00001.parquet", 5348997, "426f360d12b5d3589b66293c3417d3686a9d69eba3e82938e93d3e1545681264"),
}
HOTPOTQA_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTPOTQA_FILE = ("distractor/validation-00000-of-00001.parquet", 27452575, "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6")
MULTIHOP_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
MULTIHOP_FILES = {  # file -> (size in bytes, git blob id): the repository stores them outside LFS
    "MultiHopRAG.json": (5171312, "fcb9efe65c7730dd4126a42afb6c2c7e45721ebb"),
    "corpus.json": (6785567, "bb98345ef3921312aad05fd117b5a3e39888de2c"),
}

HF = "https://huggingface.co/datasets"

register(
    HubDataset(
        name="multihop-rag",
        title="MultiHop-RAG",
        description=(
            "609 news articles and 2,556 questions that need evidence from two to four of them: inference, comparison and temporal questions, "
            "and about 12% null queries the corpus cannot answer (abstention is exercised)."
        ),
        license="ODC-By-1.0",
        license_url="https://opendatacommons.org/licenses/by/1-0/",
        source_url="https://github.com/yixuantt/MultiHop-RAG",
        citation=(
            "Yixuan Tang and Yi Yang. 2024. MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries. arXiv:2401.15391."
        ),
        revision=MULTIHOP_REVISION,
        files=tuple(
            HubFile(name, f"{HF}/yixuantt/MultiHopRAG/resolve/{MULTIHOP_REVISION}/{name}", size, git_sha1=blob) for name, (size, blob) in MULTIHOP_FILES.items()
        ),
        reader="multihop_rag",
        shared_corpus=True,
        notes=(
            "The dataset is ODC-By (attribution); the articles in it are third-party news. RAGBench downloads them to your machine and does not "
            "redistribute them, but check the publishers' terms before you republish the corpus. A subset keeps the whole corpus: the other "
            "articles are the distractors."
        ),
    )
)

register(
    HubDataset(
        name="hotpotqa",
        title="HotpotQA (distractor, validation)",
        description=(
            "7,405 Wikipedia multi-hop questions (bridge and comparison; easy, medium, hard). Each comes with ten paragraphs: the two "
            "supporting ones and eight distractors. Supporting facts give document labels and sentence-level evidence."
        ),
        license="CC-BY-SA-4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        source_url="https://hotpotqa.github.io/",
        citation=(
            "Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov and Christopher D. Manning. 2018. "
            "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018."
        ),
        revision=HOTPOTQA_REVISION,
        files=(
            HubFile(
                "validation.parquet",
                f"{HF}/hotpotqa/hotpot_qa/resolve/{HOTPOTQA_REVISION}/{HOTPOTQA_FILE[0]}",
                HOTPOTQA_FILE[1],
                sha256=HOTPOTQA_FILE[2],
            ),
        ),
        reader="hotpotqa",
        shared_corpus=False,
        notes="Share-alike: a benchmark of this data that you publish must carry CC-BY-SA-4.0. Needs `pip install 'ragbench[hub]'` (Parquet). Use --subset: the full set is about 66,000 paragraphs.",
    )
)

for _subset, (_path, _size, _sha256) in GALILEO_TEST_FILES.items():
    register(
        HubDataset(
            name=f"galileo-{_subset}",
            title=f"RAGBench by Galileo: {_subset} (test)",
            description=(
                f"The {_subset} subset of Galileo's RAGBench (not this project): questions with the passages a RAG system was given, and "
                "annotator labels of which sentences are relevant. A question with no relevant sentence is unanswerable from its passages."
            ),
            license="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            source_url="https://huggingface.co/datasets/galileo-ai/ragbench",
            citation="Robert Friel, Masha Belyi and Atindriyo Sanyal. 2024. RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems. arXiv:2407.11005.",
            revision=GALILEO_REVISION,
            files=(HubFile("test.parquet", f"{HF}/galileo-ai/ragbench/resolve/{GALILEO_REVISION}/{_path}", _size, sha256=_sha256),),
            reader="galileo",
            shared_corpus=False,
            options={"subset": _subset},
            notes=(
                "The collection is CC-BY-4.0, but each subset is built from another dataset that carries its own terms: check the dataset card "
                "before republishing results. Reference answers are the annotated model responses the annotators found fully supported, not human "
                "gold answers. Needs `pip install 'ragbench[hub]'` (Parquet)."
            ),
        )
    )
