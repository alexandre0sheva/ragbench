# Failure Analysis

`no_failure` means the answer passed the current heuristic/LLM checks and no failure class was assigned. It is summarized below but omitted from failure-detail tables.

## Summary

| System | No Classified Failure | Classified Failures |
| --- | --- | --- |
| adaptive_default | 147 | 16 |
| agent_search_plain | 137 | 26 |
| agent_search_tools | 151 | 12 |
| bm25_default | 145 | 18 |
| contextual_default | 144 | 19 |
| corrective_default | 148 | 15 |
| decompose_default | 147 | 16 |
| full_context_ceiling | 149 | 14 |
| grep_agent_default | 141 | 22 |
| hierarchical_default | 139 | 24 |
| hybrid_default | 142 | 21 |
| hybrid_rerank_default | 145 | 18 |
| hyde_default | 148 | 15 |
| iterative_default | 149 | 14 |
| llm_heavy_default | 150 | 13 |
| no_retrieval_floor | 25 | 138 |
| parent_doc_default | 144 | 19 |
| rag_fusion_default | 147 | 16 |
| rerank_default | 148 | 15 |
| sentence_window_default | 136 | 27 |
| vector_default | 147 | 16 |
| vector_mmr | 133 | 30 |

## adaptive_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 3 | What was ClaimPilot's own net promoter score in Q3 2024? |
| bad_reranking | 2 | Which 2024 incidents lasted longer than an hour? |
| insufficient_context | 3 | What was the combined duration, in minutes, of the four 2024 Severity 1 incidents? |
| over_refusal | 2 | How long can a new starter take to finish the mandatory cybersecurity course? |
| partial_answer | 1 | What HTTP status accompanies error AR-3308? |
| possible_qrels_gap | 2 | What file types does the Broker Portal accept? |
| retrieval_miss | 3 | Compare ClaimPilot and HarborShield AI by primary workflow. |

## agent_search_plain

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 11 | How are production AI requests isolated? |
| bad_reranking | 1 | How many days separate the announcement of Maxine Thompson's award from the start of the Zurich pilot? |
| over_refusal | 8 | After how long does someone who recommended a successful hire receive their payout, and how much is it? |
| partial_answer | 3 | How long can a new starter take to finish the mandatory cybersecurity course? |
| possible_qrels_gap | 3 | Compare ClaimPilot and HarborShield AI by primary workflow. |

## agent_search_tools

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 8 | Which customer wanted renewal scenario comparison improvements? |
| over_refusal | 2 | How long can a new starter take to finish the mandatory cybersecurity course? |
| partial_answer | 1 | Which month of 2024 had the most ClaimPilot notices processed? |
| possible_qrels_gap | 1 | A ship insurer receives a stack of paperwork from an intermediary. Which tool gives a first read on the risk? |

## bm25_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 3 | What hourly rate do Premium Support engineers charge? |
| insufficient_context | 5 | How many Severity 1 incidents did RAGBench Mutual record across the first three quarters of 2024? |
| over_refusal | 1 | After how long does someone who recommended a successful hire receive their payout, and how much is it? |
| possible_qrels_gap | 3 | What file types does the Broker Portal accept? |
| retrieval_miss | 6 | When did HarborShield AI become generally available? |

## contextual_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 7 | Which compliance bulletin requires source citations for 18 months? |
| bad_reranking | 4 | How fast must the support team get back to a customer whose system is completely down? |
| insufficient_context | 6 | What was the combined duration, in minutes, of the four 2024 Severity 1 incidents? |
| possible_qrels_gap | 2 | Which tool helps a claims handler judge how serious an incoming loss report is? |

## corrective_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 5 | What was ClaimPilot's own net promoter score in Q3 2024? |
| insufficient_context | 4 | What was the combined duration, in minutes, of the four 2024 Severity 1 incidents? |
| over_refusal | 2 | What is the yearly limit for gifts and hospitality from one source? |
| partial_answer | 2 | How many days passed between HarborShield AI's general availability and the Solstice 2.1 release? |
| possible_qrels_gap | 2 | Which tool helps a claims handler judge how serious an incoming loss report is? |

## decompose_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 4 | What was ClaimPilot's own net promoter score in Q3 2024? |
| bad_reranking | 3 | Which 2024 incidents lasted longer than an hour? |
| insufficient_context | 6 | What was the total number of claim notices ClaimPilot processed in the first half of 2024? |
| possible_qrels_gap | 2 | Which tool helps a claims handler judge how serious an incoming loss report is? |
| retrieval_miss | 1 | Compare ClaimPilot and HarborShield AI by primary workflow. |

## full_context_ceiling

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 14 | When did HarborShield AI become generally available? |

## grep_agent_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 18 | Did Zurich pilot ClaimPilot? |
| bad_reranking | 1 | What file types does the Broker Portal accept? |
| possible_qrels_gap | 2 | A ship insurer receives a stack of paperwork from an intermediary. Which tool gives a first read on the risk? |
| retrieval_miss | 1 | How often is the sanctions list refreshed for the Pacific pack? |

## hierarchical_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 6 | How many Severity 1 incidents did RAGBench Mutual record across the first three quarters of 2024? |
| insufficient_context | 8 | What was the combined duration, in minutes, of the four 2024 Severity 1 incidents? |
| retrieval_miss | 10 | How long does the company keep the original files that intermediaries send in? |

## hybrid_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 5 | How was the 2024-02-14 API latency incident resolved? |
| bad_reranking | 3 | How long does the company keep the original files that intermediaries send in? |
| insufficient_context | 8 | Compare ClaimPilot and HarborShield AI by primary workflow. |
| possible_qrels_gap | 2 | Which tool helps a claims handler judge how serious an incoming loss report is? |
| retrieval_miss | 3 | How fast must the support team get back to a customer whose system is completely down? |

## hybrid_rerank_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 5 | What was ClaimPilot's own net promoter score in Q3 2024? |
| bad_reranking | 4 | Compare ClaimPilot and HarborShield AI by primary workflow. |
| insufficient_context | 3 | Who does the owner of the 2024-02-14 API latency incident report to? |
| possible_qrels_gap | 3 | What file types does the Broker Portal accept? |
| retrieval_miss | 3 | How long does the company keep the original files that intermediaries send in? |

## hyde_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 4 | How many Severity 1 incidents did RAGBench Mutual record across the first three quarters of 2024? |
| bad_reranking | 3 | Who was the owner of incident INC-2024-0630? |
| insufficient_context | 5 | Which 2024 incidents lasted longer than an hour? |
| possible_qrels_gap | 1 | A ship insurer receives a stack of paperwork from an intermediary. Which tool gives a first read on the risk? |
| retrieval_miss | 2 | What is the largest packet the platform accepts today? |

## iterative_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 7 | What runbook does Elias Ren maintain? |
| bad_reranking | 2 | What was the combined duration, in minutes, of the four 2024 Severity 1 incidents? |
| insufficient_context | 2 | Who does the owner of the 2024-02-14 API latency incident report to? |
| partial_answer | 1 | How many days did the Zurich pilot last? |
| possible_qrels_gap | 2 | Which tool helps a claims handler judge how serious an incoming loss report is? |

## llm_heavy_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 3 | Did Zurich pilot ClaimPilot? |
| bad_reranking | 3 | What was the combined duration, in minutes, of the four 2024 Severity 1 incidents? |
| insufficient_context | 4 | Who does the owner of the 2024-02-14 API latency incident report to? |
| possible_qrels_gap | 1 | A ship insurer receives a stack of paperwork from an intermediary. Which tool gives a first read on the risk? |
| retrieval_miss | 2 | How long does the company keep the original files that intermediaries send in? |

## no_retrieval_floor

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 9 | What evidence should support attach for AI answer quality escalations? |
| over_refusal | 129 | Who won the prestigious IIOTY award in 2023? |

## parent_doc_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 7 | Which tool helps a claims handler judge how serious an incoming loss report is? |
| bad_reranking | 3 | What was the total number of claim notices ClaimPilot processed in the first half of 2024? |
| insufficient_context | 6 | How many Severity 1 incidents did RAGBench Mutual record across the first three quarters of 2024? |
| possible_qrels_gap | 2 | Which product should a marine underwriter use for a first-pass risk memo? |
| retrieval_miss | 1 | What is the largest packet the platform accepts today? |

## rag_fusion_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 4 | What was ClaimPilot's own net promoter score in Q3 2024? |
| bad_reranking | 3 | What is the largest packet the platform accepts today? |
| insufficient_context | 6 | Which 2024 incidents lasted longer than an hour? |
| possible_qrels_gap | 2 | Which tool helps a claims handler judge how serious an incoming loss report is? |
| run_error | 1 | Does HarborShield AI's CSV export include the sanctions rationale in the latest release? |

## rerank_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 6 | How many days passed between HarborShield AI's general availability and the Solstice 2.1 release? |
| bad_reranking | 3 | What is the largest packet the platform accepts today? |
| insufficient_context | 3 | How many Severity 1 incidents did RAGBench Mutual record across the first three quarters of 2024? |
| possible_qrels_gap | 1 | A ship insurer receives a stack of paperwork from an intermediary. Which tool gives a first read on the risk? |
| retrieval_miss | 2 | How long does the company keep the original files that intermediaries send in? |

## sentence_window_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 9 | Which customer wanted renewal scenario comparison improvements? |
| bad_reranking | 3 | Who was the owner of incident INC-2024-0630? |
| insufficient_context | 5 | What was the combined duration, in minutes, of the four 2024 Severity 1 incidents? |
| over_refusal | 7 | How was the 2024-02-14 API latency incident resolved? |
| partial_answer | 1 | Which is cheaper per month, the Mediterranean or the Baltic pack, and by how much? |
| possible_qrels_gap | 1 | Compare ClaimPilot and HarborShield AI by primary workflow. |
| retrieval_miss | 1 | Which customers in the contract register use ClaimPilot? |

## vector_default

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 4 | How many Severity 1 incidents did RAGBench Mutual record across the first three quarters of 2024? |
| bad_reranking | 2 | What was the combined duration, in minutes, of the four 2024 Severity 1 incidents? |
| insufficient_context | 6 | What was the total number of claim notices ClaimPilot processed in the first half of 2024? |
| over_refusal | 1 | By what percentage did Lakeshore Mutual's average first-touch time fall? |
| possible_qrels_gap | 2 | Which product should a marine underwriter use for a first-pass risk memo? |
| retrieval_miss | 1 | What is the largest packet the platform accepts today? |

## vector_mmr

| Failure Type | Count | Example Question |
| --- | --- | --- |
| answer_hallucination | 4 | What was the total number of claim notices ClaimPilot processed in the first half of 2024? |
| bad_reranking | 2 | What is the largest packet the platform accepts today? |
| insufficient_context | 17 | By what percentage did quarterly ClaimPilot notice volume grow from Q1 to Q3 2024? |
| over_refusal | 3 | How fast must the support team get back to a customer whose system is completely down? |
| possible_qrels_gap | 3 | What products are covered by Compliance Bulletin 2023-11? |
| retrieval_miss | 1 | How many more HarborShield packets were reviewed in Q3 2024 than in Q1 2024? |
