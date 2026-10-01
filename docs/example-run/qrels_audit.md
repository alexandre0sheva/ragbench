# Qrels Audit

This report lists cases where the answer judge rated an answer highly even though retrieval did not find all labeled relevant documents. These are candidates for reviewing qrels, not automatic qrel changes.

| System | Question | Severity | Recall@5 | Labeled Docs | Unlabeled Retrieved Docs |
| --- | --- | --- | --- | --- | --- |
| bm25_default | q_023 | high | 0.00 | doc_018 | doc_026, doc_048, doc_049 |
| bm25_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_036, doc_003, doc_049 |
| bm25_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_004, doc_055, doc_028, doc_009 |
| bm25_default | q_046 | high | 0.00 | doc_003, doc_021 | doc_049, doc_023 |
| bm25_default | q_047 | high | 0.00 | doc_002 | doc_048, doc_050 |
| bm25_default | q_084 | medium | 0.50 | doc_037, doc_026 | doc_048, doc_018 |
| bm25_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_048, doc_006, doc_047 |
| bm25_default | q_088 | medium | 0.50 | doc_036, doc_027 | doc_025, doc_048, doc_049, doc_018 |
| bm25_default | q_092 | medium | 0.50 | doc_028, doc_012 | doc_030, doc_049, doc_045, doc_034 |
| bm25_default | q_095 | medium | 0.50 | doc_055, doc_042 | doc_047, doc_028 |
| bm25_default | q_105 | medium | 0.50 | doc_011, doc_009 | doc_030, doc_015, doc_036, doc_037 |
| bm25_default | q_120 | medium | 0.50 | doc_006, doc_036 | doc_019, doc_058, doc_049, doc_027 |
| bm25_default | q_123 | medium | 0.67 | doc_055, doc_040, doc_025 | doc_041, doc_042, doc_048 |
| bm25_default | q_124 | medium | 0.50 | doc_014, doc_057 | doc_005, doc_049, doc_047 |
| vector_default | q_013 | medium | 0.50 | doc_009, doc_018 | doc_037, doc_002, doc_045, doc_046 |
| vector_default | q_025 | medium | 0.50 | doc_022, doc_002 | doc_004, doc_041, doc_040, doc_055 |
| vector_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_002, doc_048, doc_017 |
| vector_default | q_060 | medium | 0.50 | doc_025, doc_026 | doc_052, doc_011, doc_051, doc_038 |
| vector_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_021, doc_003, doc_049 |
| vector_default | q_088 | medium | 0.50 | doc_036, doc_027 | doc_049, doc_003 |
| vector_default | q_123 | medium | 0.67 | doc_055, doc_040, doc_025 | doc_042, doc_041, doc_049 |
| hybrid_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_048, doc_037 |
| hybrid_default | q_046 | medium | 0.50 | doc_003, doc_021 | doc_049, doc_023 |
| hybrid_default | q_047 | high | 0.00 | doc_002 | doc_048 |
| hybrid_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_006, doc_013, doc_049 |
| hybrid_default | q_124 | medium | 0.50 | doc_014, doc_057 | doc_047 |
| rerank_default | q_026 | medium | 0.50 | doc_002, doc_003 | doc_050, doc_022, doc_017, doc_012 |
| rerank_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_003, doc_049, doc_036 |
| rerank_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_037, doc_048 |
| rerank_default | q_046 | medium | 0.50 | doc_003, doc_021 | doc_049 |
| rerank_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_049 |
| rerank_default | q_095 | medium | 0.50 | doc_055, doc_042 | doc_047, doc_028, doc_012 |
| rerank_default | q_105 | medium | 0.50 | doc_011, doc_009 | doc_015, doc_030, doc_050 |
| rerank_default | q_123 | medium | 0.67 | doc_055, doc_040, doc_025 | doc_042, doc_041, doc_048 |
| hybrid_rerank_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_003, doc_049, doc_036 |
| hybrid_rerank_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_009, doc_048, doc_001 |
| hybrid_rerank_default | q_046 | high | 0.00 | doc_003, doc_021 | doc_049, doc_050 |
| hybrid_rerank_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_048, doc_049, doc_047 |
| hybrid_rerank_default | q_088 | medium | 0.50 | doc_036, doc_027 | doc_025, doc_049 |
| hybrid_rerank_default | q_095 | medium | 0.50 | doc_055, doc_042 | doc_047, doc_028 |
| hybrid_rerank_default | q_105 | medium | 0.50 | doc_011, doc_009 | doc_015, doc_030, doc_050 |
| hybrid_rerank_default | q_109 | medium | 0.67 | doc_030, doc_031, doc_032 | doc_050, doc_001 |
| hybrid_rerank_default | q_120 | medium | 0.50 | doc_006, doc_036 | doc_049, doc_019, doc_058, doc_027 |
| hybrid_rerank_default | q_123 | medium | 0.67 | doc_055, doc_040, doc_025 | doc_042, doc_041, doc_048 |
| hybrid_rerank_default | q_124 | medium | 0.50 | doc_014, doc_057 | doc_005, doc_047, doc_050 |
| hyde_default | q_025 | medium | 0.50 | doc_022, doc_002 | doc_004, doc_055, doc_041, doc_040 |
| hyde_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_003, doc_036, doc_049 |
| hyde_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_002, doc_048, doc_017 |
| hyde_default | q_078 | medium | 0.50 | doc_040, doc_028 | doc_042, doc_041, doc_055, doc_012 |
| hyde_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_021, doc_003, doc_049 |
| hyde_default | q_088 | medium | 0.50 | doc_036, doc_027 | doc_003, doc_049, doc_021 |
| hyde_default | q_123 | medium | 0.67 | doc_055, doc_040, doc_025 | doc_042, doc_041, doc_053 |
| parent_doc_default | q_013 | medium | 0.50 | doc_009, doc_018 | doc_037, doc_002, doc_045 |
| parent_doc_default | q_025 | medium | 0.50 | doc_022, doc_002 | doc_004, doc_041, doc_040 |
| parent_doc_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_002, doc_017 |
| parent_doc_default | q_060 | medium | 0.50 | doc_025, doc_026 | doc_052, doc_048, doc_011 |
| parent_doc_default | q_083 | medium | 0.50 | doc_037, doc_026 | doc_018, doc_011, doc_017 |
| parent_doc_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_021, doc_003, doc_049 |
| llm_heavy_default | q_026 | medium | 0.50 | doc_002, doc_003 | doc_017, doc_022, doc_012, doc_031 |
| llm_heavy_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_003, doc_049 |
| llm_heavy_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_049, doc_048, doc_002 |
| llm_heavy_default | q_046 | medium | 0.50 | doc_003, doc_021 | doc_049 |
| llm_heavy_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_049, doc_003 |
| llm_heavy_default | q_105 | medium | 0.50 | doc_011, doc_009 | doc_015, doc_030, doc_050, doc_051 |
| llm_heavy_default | q_124 | medium | 0.50 | doc_014, doc_057 | doc_005, doc_047, doc_050, doc_048 |
| full_context_ceiling | q_003 | high | 0.00 | doc_002 | doc_018, doc_048, doc_016, doc_030, doc_008, doc_049, doc_006, doc_056, doc_040, doc_046, doc_043, doc_044, doc_045, doc_017, doc_014, doc_052, doc_010, doc_029, doc_022, doc_037, doc_019, doc_007, doc_005, doc_012, doc_060, doc_031, doc_050, doc_011, doc_001, doc_058, doc_057, doc_038, doc_039, doc_051, doc_028, doc_047, doc_009, doc_025, doc_020, doc_013, doc_032, doc_026, doc_024, doc_003, doc_004, doc_015, doc_021, doc_023, doc_027, doc_033, doc_034, doc_035, doc_036, doc_041, doc_042, doc_053, doc_054, doc_055, doc_059 |
| full_context_ceiling | q_023 | high | 0.00 | doc_018 | doc_026, doc_048, doc_024, doc_049, doc_023, doc_009, doc_025, doc_050, doc_047, doc_037, doc_015, doc_002, doc_011, doc_022, doc_017, doc_051, doc_003, doc_027, doc_033, doc_046, doc_043, doc_044, doc_045, doc_038, doc_042, doc_041, doc_040, doc_039, doc_007, doc_029, doc_005, doc_053, doc_036, doc_052, doc_032, doc_060, doc_014, doc_058, doc_019, doc_055, doc_057, doc_030, doc_034, doc_028, doc_035, doc_054, doc_021, doc_020, doc_008, doc_013, doc_012, doc_010, doc_006, doc_001, doc_016, doc_004, doc_031, doc_059, doc_056 |
| full_context_ceiling | q_025 | medium | 0.50 | doc_022, doc_002 | doc_055, doc_057, doc_029, doc_048, doc_026, doc_024, doc_050, doc_051, doc_047, doc_058, doc_027, doc_028, doc_031, doc_020, doc_049, doc_019, doc_059, doc_010, doc_001, doc_060, doc_053, doc_038, doc_039, doc_023, doc_025, doc_037, doc_009, doc_036, doc_013, doc_003, doc_004, doc_054, doc_034, doc_035, doc_052, doc_056, doc_041, doc_042, doc_040, doc_011, doc_006, doc_012, doc_008, doc_017, doc_033, doc_030, doc_032, doc_005, doc_007, doc_014, doc_015, doc_016, doc_018, doc_021, doc_043, doc_044, doc_045, doc_046 |
| full_context_ceiling | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_037, doc_004, doc_028, doc_001, doc_009, doc_055, doc_057, doc_049, doc_048, doc_014, doc_046, doc_043, doc_044, doc_045, doc_002, doc_005, doc_019, doc_018, doc_011, doc_047, doc_017, doc_013, doc_007, doc_051, doc_012, doc_058, doc_050, doc_025, doc_022, doc_038, doc_056, doc_039, doc_031, doc_024, doc_053, doc_036, doc_023, doc_033, doc_027, doc_052, doc_015, doc_032, doc_029, doc_026, doc_042, doc_030, doc_034, doc_035, doc_054, doc_041, doc_021, doc_040, doc_020, doc_008, doc_006, doc_016, doc_059 |
| full_context_ceiling | q_046 | high | 0.00 | doc_003, doc_021 | doc_049, doc_048, doc_023, doc_050, doc_047, doc_052, doc_025, doc_054, doc_051, doc_024, doc_026, doc_001, doc_033, doc_027, doc_053, doc_004, doc_031, doc_029, doc_035, doc_034, doc_040, doc_011, doc_041, doc_042, doc_006, doc_058, doc_032, doc_012, doc_002, doc_060, doc_055, doc_038, doc_030, doc_010, doc_037, doc_036, doc_057, doc_018, doc_015, doc_022, doc_019, doc_014, doc_016, doc_007, doc_009, doc_005, doc_046, doc_043, doc_044, doc_045, doc_020, doc_013, doc_056, doc_028, doc_008, doc_017, doc_039, doc_059 |
| full_context_ceiling | q_047 | high | 0.00 | doc_002 | doc_048, doc_049, doc_050, doc_047, doc_024, doc_058, doc_029, doc_023, doc_038, doc_034, doc_051, doc_027, doc_054, doc_026, doc_037, doc_036, doc_035, doc_025, doc_006, doc_033, doc_052, doc_055, doc_011, doc_040, doc_012, doc_053, doc_042, doc_041, doc_028, doc_022, doc_007, doc_032, doc_001, doc_004, doc_019, doc_003, doc_031, doc_020, doc_060, doc_016, doc_005, doc_008, doc_030, doc_009, doc_013, doc_059, doc_057, doc_046, doc_043, doc_044, doc_045, doc_015, doc_014, doc_010, doc_018, doc_021, doc_017, doc_039, doc_056 |
| full_context_ceiling | q_050 | high | 0.00 | doc_016 | doc_023, doc_047, doc_048, doc_033, doc_049, doc_054, doc_050, doc_024, doc_034, doc_035, doc_021, doc_003, doc_025, doc_009, doc_001, doc_059, doc_005, doc_038, doc_060, doc_026, doc_037, doc_014, doc_052, doc_055, doc_020, doc_027, doc_039, doc_006, doc_057, doc_028, doc_007, doc_018, doc_032, doc_051, doc_030, doc_036, doc_017, doc_042, doc_046, doc_043, doc_044, doc_045, doc_053, doc_041, doc_040, doc_010, doc_015, doc_058, doc_011, doc_029, doc_019, doc_031, doc_008, doc_013, doc_004, doc_012, doc_002, doc_022, doc_056 |
| full_context_ceiling | q_085 | medium | 0.50 | doc_036, doc_027 | doc_048, doc_047, doc_006, doc_049, doc_053, doc_054, doc_013, doc_050, doc_024, doc_025, doc_023, doc_028, doc_009, doc_003, doc_038, doc_039, doc_033, doc_046, doc_043, doc_044, doc_045, doc_041, doc_042, doc_040, doc_029, doc_001, doc_016, doc_019, doc_015, doc_002, doc_032, doc_060, doc_037, doc_008, doc_004, doc_017, doc_014, doc_007, doc_005, doc_012, doc_010, doc_026, doc_055, doc_030, doc_031, doc_034, doc_058, doc_035, doc_051, doc_057, doc_021, doc_022, doc_052, doc_056, doc_011, doc_018, doc_020, doc_059 |
| full_context_ceiling | q_088 | medium | 0.50 | doc_036, doc_027 | doc_025, doc_049, doc_042, doc_041, doc_040, doc_018, doc_026, doc_048, doc_050, doc_023, doc_047, doc_046, doc_043, doc_044, doc_045, doc_012, doc_024, doc_037, doc_021, doc_028, doc_014, doc_056, doc_052, doc_038, doc_003, doc_058, doc_005, doc_015, doc_009, doc_053, doc_051, doc_033, doc_055, doc_002, doc_001, doc_032, doc_060, doc_010, doc_039, doc_011, doc_029, doc_019, doc_057, doc_030, doc_034, doc_035, doc_054, doc_020, doc_008, doc_013, doc_007, doc_006, doc_022, doc_004, doc_017, doc_031, doc_059, doc_016 |
| full_context_ceiling | q_092 | medium | 0.50 | doc_028, doc_012 | doc_045, doc_030, doc_034, doc_049, doc_047, doc_023, doc_048, doc_046, doc_043, doc_044, doc_036, doc_041, doc_042, doc_040, doc_031, doc_035, doc_025, doc_052, doc_014, doc_050, doc_059, doc_032, doc_005, doc_003, doc_037, doc_058, doc_029, doc_027, doc_024, doc_026, doc_051, doc_053, doc_039, doc_010, doc_038, doc_060, doc_001, doc_006, doc_008, doc_009, doc_017, doc_019, doc_021, doc_022, doc_033, doc_054, doc_055, doc_056, doc_057, doc_002, doc_004, doc_007, doc_011, doc_013, doc_015, doc_016, doc_018, doc_020 |
| full_context_ceiling | q_093 | medium | 0.50 | doc_041, doc_028 | doc_040, doc_042, doc_044, doc_049, doc_048, doc_012, doc_047, doc_052, doc_055, doc_024, doc_038, doc_050, doc_046, doc_043, doc_045, doc_035, doc_023, doc_032, doc_030, doc_036, doc_031, doc_056, doc_013, doc_060, doc_039, doc_025, doc_027, doc_029, doc_018, doc_058, doc_003, doc_009, doc_005, doc_015, doc_014, doc_033, doc_026, doc_002, doc_001, doc_053, doc_037, doc_019, doc_051, doc_057, doc_011, doc_020, doc_008, doc_010, doc_034, doc_054, doc_021, doc_007, doc_006, doc_022, doc_004, doc_017, doc_059, doc_016 |
| full_context_ceiling | q_101 | medium | 0.50 | doc_002, doc_009 | doc_037, doc_036, doc_039, doc_038, doc_048, doc_050, doc_023, doc_059, doc_003, doc_030, doc_047, doc_018, doc_004, doc_011, doc_049, doc_035, doc_034, doc_025, doc_051, doc_056, doc_033, doc_060, doc_054, doc_055, doc_028, doc_046, doc_043, doc_044, doc_045, doc_031, doc_053, doc_014, doc_010, doc_024, doc_017, doc_005, doc_001, doc_057, doc_019, doc_026, doc_022, doc_007, doc_012, doc_013, doc_058, doc_029, doc_027, doc_020, doc_042, doc_041, doc_040, doc_032, doc_015, doc_052, doc_008, doc_016, doc_021, doc_006 |
| full_context_ceiling | q_105 | medium | 0.50 | doc_011, doc_009 | doc_030, doc_015, doc_050, doc_036, doc_051, doc_037, doc_052, doc_059, doc_025, doc_047, doc_054, doc_026, doc_056, doc_053, doc_035, doc_034, doc_055, doc_049, doc_032, doc_048, doc_003, doc_033, doc_031, doc_023, doc_058, doc_018, doc_028, doc_038, doc_004, doc_060, doc_013, doc_039, doc_024, doc_020, doc_027, doc_019, doc_008, doc_012, doc_007, doc_006, doc_002, doc_057, doc_005, doc_046, doc_043, doc_044, doc_045, doc_014, doc_016, doc_029, doc_042, doc_041, doc_021, doc_040, doc_010, doc_001, doc_022, doc_017 |
| full_context_ceiling | q_109 | medium | 0.33 | doc_030, doc_031, doc_032 | doc_036, doc_034, doc_050, doc_035, doc_006, doc_049, doc_023, doc_054, doc_001, doc_047, doc_055, doc_008, doc_038, doc_048, doc_039, doc_033, doc_004, doc_025, doc_024, doc_057, doc_026, doc_037, doc_005, doc_014, doc_053, doc_009, doc_052, doc_051, doc_028, doc_029, doc_056, doc_027, doc_058, doc_019, doc_060, doc_011, doc_013, doc_002, doc_020, doc_012, doc_015, doc_059, doc_003, doc_021, doc_016, doc_046, doc_043, doc_044, doc_045, doc_018, doc_042, doc_041, doc_040, doc_007, doc_010, doc_022, doc_017 |
| full_context_ceiling | q_110 | medium | 0.75 | doc_011, doc_051, doc_052, doc_053 | doc_050, doc_032, doc_030, doc_024, doc_033, doc_031, doc_049, doc_036, doc_054, doc_047, doc_057, doc_048, doc_058, doc_006, doc_009, doc_038, doc_020, doc_003, doc_013, doc_039, doc_027, doc_060, doc_025, doc_001, doc_037, doc_023, doc_028, doc_007, doc_026, doc_055, doc_046, doc_043, doc_044, doc_045, doc_034, doc_035, doc_042, doc_019, doc_014, doc_041, doc_005, doc_040, doc_015, doc_008, doc_018, doc_012, doc_021, doc_029, doc_002, doc_059, doc_017, doc_056, doc_010, doc_022, doc_004, doc_016 |
| full_context_ceiling | q_115 | high | 0.00 | doc_011, doc_051, doc_052, doc_053 | doc_015, doc_031, doc_032, doc_030, doc_050, doc_036, doc_049, doc_057, doc_002, doc_047, doc_033, doc_001, doc_024, doc_038, doc_009, doc_039, doc_048, doc_058, doc_037, doc_019, doc_060, doc_013, doc_020, doc_008, doc_012, doc_059, doc_003, doc_021, doc_028, doc_055, doc_056, doc_029, doc_054, doc_006, doc_034, doc_035, doc_026, doc_025, doc_005, doc_046, doc_043, doc_044, doc_045, doc_023, doc_027, doc_014, doc_018, doc_042, doc_041, doc_040, doc_007, doc_010, doc_022, doc_004, doc_017, doc_016 |
| full_context_ceiling | q_116 | medium | 0.50 | doc_011, doc_057 | doc_030, doc_051, doc_015, doc_050, doc_052, doc_053, doc_037, doc_047, doc_036, doc_048, doc_009, doc_049, doc_032, doc_026, doc_025, doc_031, doc_054, doc_059, doc_024, doc_028, doc_038, doc_039, doc_058, doc_033, doc_002, doc_056, doc_023, doc_060, doc_055, doc_020, doc_034, doc_035, doc_003, doc_013, doc_006, doc_029, doc_019, doc_027, doc_008, doc_012, doc_010, doc_001, doc_022, doc_005, doc_046, doc_043, doc_044, doc_045, doc_014, doc_018, doc_042, doc_041, doc_021, doc_040, doc_007, doc_004, doc_017, doc_016 |
| full_context_ceiling | q_119 | medium | 0.50 | doc_055, doc_056 | doc_050, doc_049, doc_047, doc_048, doc_039, doc_035, doc_034, doc_033, doc_024, doc_016, doc_051, doc_032, doc_038, doc_054, doc_060, doc_023, doc_028, doc_036, doc_001, doc_030, doc_027, doc_037, doc_052, doc_009, doc_025, doc_031, doc_006, doc_003, doc_058, doc_007, doc_021, doc_057, doc_026, doc_010, doc_053, doc_011, doc_046, doc_043, doc_044, doc_045, doc_005, doc_002, doc_029, doc_022, doc_015, doc_019, doc_020, doc_042, doc_013, doc_059, doc_014, doc_041, doc_040, doc_008, doc_004, doc_012, doc_017, doc_018 |
| full_context_ceiling | q_120 | medium | 0.50 | doc_006, doc_036 | doc_027, doc_019, doc_058, doc_049, doc_026, doc_029, doc_050, doc_048, doc_047, doc_054, doc_051, doc_031, doc_008, doc_007, doc_033, doc_024, doc_034, doc_035, doc_030, doc_055, doc_023, doc_025, doc_003, doc_037, doc_028, doc_005, doc_002, doc_060, doc_052, doc_014, doc_057, doc_032, doc_009, doc_020, doc_017, doc_012, doc_001, doc_004, doc_015, doc_013, doc_046, doc_043, doc_044, doc_045, doc_010, doc_018, doc_053, doc_038, doc_042, doc_041, doc_040, doc_039, doc_016, doc_011, doc_021, doc_022, doc_059, doc_056 |
| full_context_ceiling | q_123 | medium | 0.67 | doc_055, doc_040, doc_025 | doc_041, doc_042, doc_048, doc_050, doc_049, doc_047, doc_008, doc_024, doc_023, doc_056, doc_022, doc_026, doc_028, doc_030, doc_020, doc_046, doc_043, doc_044, doc_045, doc_012, doc_029, doc_054, doc_032, doc_006, doc_027, doc_002, doc_037, doc_052, doc_007, doc_038, doc_003, doc_057, doc_051, doc_058, doc_005, doc_011, doc_001, doc_031, doc_009, doc_034, doc_033, doc_036, doc_053, doc_004, doc_039, doc_059, doc_015, doc_060, doc_013, doc_010, doc_014, doc_035, doc_019, doc_018, doc_021, doc_017, doc_016 |
| full_context_ceiling | q_124 | medium | 0.50 | doc_014, doc_057 | doc_005, doc_047, doc_049, doc_048, doc_050, doc_002, doc_051, doc_023, doc_001, doc_052, doc_009, doc_024, doc_011, doc_025, doc_038, doc_003, doc_053, doc_039, doc_033, doc_027, doc_029, doc_010, doc_036, doc_060, doc_037, doc_055, doc_032, doc_026, doc_030, doc_031, doc_020, doc_006, doc_016, doc_034, doc_035, doc_028, doc_054, doc_022, doc_059, doc_046, doc_043, doc_044, doc_045, doc_015, doc_058, doc_042, doc_041, doc_040, doc_008, doc_013, doc_004, doc_019, doc_018, doc_021, doc_017, doc_007, doc_012, doc_056 |
| sentence_window_default | q_006 | medium | 0.50 | doc_006, doc_019 | doc_049, doc_058, doc_036, doc_027 |
| sentence_window_default | q_026 | high | 0.00 | doc_002, doc_003 | doc_047, doc_001, doc_060, doc_048, doc_050 |
| sentence_window_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_003, doc_036, doc_027 |
| sentence_window_default | q_032 | medium | 0.50 | doc_020, doc_012 | doc_037, doc_010, doc_002, doc_048 |
| sentence_window_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_048, doc_060, doc_002 |
| sentence_window_default | q_111 | medium | 0.75 | doc_051, doc_053, doc_011, doc_052 | doc_031, doc_050 |
| sentence_window_default | q_124 | medium | 0.50 | doc_014, doc_057 | doc_005, doc_050, doc_047, doc_048 |
| vector_mmr | q_001 | medium | 0.50 | doc_005, doc_014 | doc_059, doc_016, doc_057, doc_035 |
| vector_mmr | q_015 | high | 0.00 | doc_010 | doc_060, doc_047, doc_048, doc_017, doc_025 |
| vector_mmr | q_016 | medium | 0.50 | doc_011, doc_015 | doc_038, doc_050, doc_052, doc_030 |
| vector_mmr | q_025 | medium | 0.50 | doc_022, doc_002 | doc_004, doc_058, doc_048, doc_042 |
| vector_mmr | q_026 | medium | 0.50 | doc_002, doc_003 | doc_022, doc_007, doc_031, doc_012 |
| vector_mmr | q_028 | medium | 0.50 | doc_009, doc_021 | doc_003, doc_036, doc_012, doc_006 |
| vector_mmr | q_037 | medium | 0.50 | doc_010, doc_003 | doc_013, doc_017, doc_051, doc_048 |
| vector_mmr | q_038 | high | 0.00 | doc_002 | doc_048, doc_054, doc_007, doc_004 |
| vector_mmr | q_046 | medium | 0.50 | doc_003, doc_021 | doc_048, doc_049, doc_054, doc_013 |
| vector_mmr | q_047 | high | 0.00 | doc_002 | doc_048, doc_021, doc_007, doc_023, doc_004 |
| vector_mmr | q_051 | medium | 0.50 | doc_023, doc_049 | doc_051, doc_016, doc_056, doc_054 |
| vector_mmr | q_058 | medium | 0.50 | doc_025, doc_048 | doc_053, doc_013, doc_007, doc_051 |
| vector_mmr | q_060 | medium | 0.50 | doc_025, doc_026 | doc_052, doc_011, doc_048, doc_013 |
| vector_mmr | q_078 | medium | 0.50 | doc_040, doc_028 | doc_017, doc_020, doc_012, doc_055 |
| vector_mmr | q_083 | medium | 0.50 | doc_037, doc_026 | doc_011, doc_048, doc_018, doc_012 |
| vector_mmr | q_085 | medium | 0.50 | doc_036, doc_027 | doc_013, doc_021, doc_054, doc_012 |
| vector_mmr | q_088 | medium | 0.50 | doc_036, doc_027 | doc_049, doc_056, doc_012, doc_008 |
| vector_mmr | q_094 | medium | 0.50 | doc_042, doc_041 | doc_028, doc_055, doc_020, doc_012 |
| vector_mmr | q_095 | medium | 0.50 | doc_055, doc_042 | doc_028, doc_044, doc_041, doc_047 |
| vector_mmr | q_101 | medium | 0.50 | doc_002, doc_009 | doc_059, doc_045, doc_037, doc_039 |
| vector_mmr | q_123 | medium | 0.67 | doc_055, doc_040, doc_025 | doc_053, doc_049, doc_037 |
| vector_mmr | q_125 | medium | 0.50 | doc_040, doc_042 | doc_016, doc_020, doc_055, doc_039 |
| contextual_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_003, doc_036, doc_049, doc_030 |
| contextual_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_017, doc_037, doc_001 |
| contextual_default | q_046 | high | 0.00 | doc_003, doc_021 | doc_049, doc_023 |
| contextual_default | q_047 | high | 0.00 | doc_002 | doc_048 |
| contextual_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_006, doc_049, doc_048, doc_054 |
| contextual_default | q_123 | medium | 0.67 | doc_055, doc_040, doc_025 | doc_041, doc_042, doc_048 |
| hierarchical_default | q_026 | medium | 0.50 | doc_002, doc_003 | doc_022, doc_017, doc_031, doc_037 |
| hierarchical_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_003, doc_036, doc_054, doc_030 |
| hierarchical_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_002, doc_004, doc_009 |
| hierarchical_default | q_058 | medium | 0.50 | doc_025, doc_048 | doc_042, doc_055, doc_040 |
| hierarchical_default | q_060 | medium | 0.50 | doc_025, doc_026 | doc_052, doc_017, doc_027 |
| hierarchical_default | q_103 | medium | 0.50 | doc_059, doc_050 | doc_060, doc_051, doc_034, doc_035 |
| hierarchical_default | q_113 | medium | 0.67 | doc_041, doc_042, doc_040 | doc_028, doc_017 |
| hierarchical_default | q_122 | medium | 0.50 | doc_027, doc_036 | doc_019, doc_006, doc_054, doc_058 |
| hierarchical_default | q_123 | medium | 0.67 | doc_055, doc_040, doc_025 | doc_042, doc_041, doc_052 |
| hierarchical_default | q_124 | medium | 0.50 | doc_014, doc_057 | doc_005, doc_053, doc_008, doc_001 |
| rag_fusion_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_048, doc_016 |
| rag_fusion_default | q_046 | medium | 0.50 | doc_003, doc_021 | doc_049 |
| rag_fusion_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_049, doc_048, doc_003 |
| rag_fusion_default | q_088 | medium | 0.50 | doc_036, doc_027 | doc_049 |
| decompose_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_003, doc_036, doc_049 |
| decompose_default | q_030 | medium | 0.50 | doc_006, doc_019 | doc_036, doc_026, doc_058, doc_009 |
| decompose_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_048, doc_037, doc_047 |
| decompose_default | q_046 | high | 0.00 | doc_003, doc_021 | doc_049 |
| decompose_default | q_047 | high | 0.00 | doc_002 | doc_048 |
| decompose_default | q_084 | medium | 0.50 | doc_037, doc_026 | doc_018, doc_048, doc_025, doc_043 |
| decompose_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_006, doc_049, doc_013, doc_003 |
| decompose_default | q_088 | medium | 0.50 | doc_036, doc_027 | doc_049 |
| decompose_default | q_101 | medium | 0.50 | doc_002, doc_009 | doc_024, doc_037, doc_059, doc_030 |
| decompose_default | q_105 | medium | 0.50 | doc_011, doc_009 | doc_059, doc_030, doc_037, doc_036 |
| decompose_default | q_115 | medium | 0.50 | doc_011, doc_051, doc_052, doc_053 | doc_050 |
| corrective_default | q_006 | medium | 0.50 | doc_006, doc_019 | doc_058, doc_036, doc_049 |
| corrective_default | q_026 | medium | 0.50 | doc_002, doc_003 | doc_022, doc_010, doc_019, doc_004 |
| corrective_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_003, doc_036, doc_049 |
| corrective_default | q_029 | medium | 0.50 | doc_008, doc_019 | doc_055, doc_041, doc_042, doc_040 |
| corrective_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_009, doc_002, doc_055 |
| corrective_default | q_046 | medium | 0.50 | doc_003, doc_021 | doc_049, doc_023 |
| corrective_default | q_047 | high | 0.00 | doc_002 | doc_048 |
| corrective_default | q_051 | medium | 0.50 | doc_023, doc_049 | doc_024 |
| corrective_default | q_058 | medium | 0.50 | doc_025, doc_048 | doc_026, doc_051, doc_041, doc_011 |
| corrective_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_049, doc_054, doc_048 |
| corrective_default | q_088 | medium | 0.50 | doc_036, doc_027 | doc_049, doc_012, doc_025 |
| corrective_default | q_095 | medium | 0.50 | doc_055, doc_042 | doc_028, doc_012, doc_056, doc_047 |
| iterative_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_036, doc_003, doc_030, doc_049 |
| iterative_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_048, doc_047 |
| iterative_default | q_046 | medium | 0.50 | doc_003, doc_021 | doc_049 |
| iterative_default | q_047 | high | 0.00 | doc_002 | doc_048 |
| iterative_default | q_084 | medium | 0.50 | doc_037, doc_026 | doc_048 |
| iterative_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_006, doc_049, doc_013, doc_003 |
| iterative_default | q_088 | medium | 0.50 | doc_036, doc_027 | doc_049 |
| iterative_default | q_105 | medium | 0.50 | doc_011, doc_009 | doc_030, doc_037, doc_059, doc_036 |
| iterative_default | q_111 | medium | 0.25 | doc_051, doc_053, doc_011, doc_052 | doc_050, doc_031, doc_032, doc_030 |
| agent_search_plain | q_026 | medium | 0.50 | doc_002, doc_003 | doc_048, doc_049 |
| agent_search_plain | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_048, doc_001 |
| agent_search_plain | q_046 | medium | 0.50 | doc_003, doc_021 | doc_049 |
| agent_search_plain | q_049 | high | 0.00 | doc_013 | doc_038, doc_039, doc_047, doc_050 |
| agent_search_plain | q_085 | medium | 0.50 | doc_036, doc_027 | doc_006, doc_048, doc_013, doc_049 |
| agent_search_plain | q_088 | medium | 0.50 | doc_036, doc_027 | doc_049 |
| agent_search_plain | q_111 | medium | 0.75 | doc_051, doc_053, doc_011, doc_052 | doc_050 |
| agent_search_plain | q_118 | medium | 0.50 | doc_053, doc_039 | doc_038, doc_032, doc_012, doc_028 |
| agent_search_tools | q_047 | high | 0.00 | doc_002 | doc_048 |
| agent_search_tools | q_085 | medium | 0.50 | doc_036, doc_027 | doc_006, doc_048, doc_049 |
| agent_search_tools | q_088 | medium | 0.50 | doc_036, doc_027 | doc_049, doc_021 |
| agent_search_tools | q_110 | medium | 0.75 | doc_011, doc_051, doc_052, doc_053 | doc_050, doc_030 |
| agent_search_tools | q_111 | medium | 0.50 | doc_051, doc_053, doc_011, doc_052 | doc_050, doc_030, doc_032 |
| agent_search_tools | q_117 | medium | 0.50 | doc_007, doc_057 | doc_002, doc_020, doc_058, doc_048 |
| agent_search_tools | q_118 | medium | 0.50 | doc_053, doc_039 | doc_038, doc_032, doc_012 |
| agent_search_tools | q_130 | medium | 0.50 | doc_053, doc_051 | doc_031, doc_052, doc_032, doc_025 |
| grep_agent_default | q_047 | high | 0.00 | doc_002 | doc_048, doc_009, doc_027, doc_031 |
| grep_agent_default | q_051 | medium | 0.50 | doc_023, doc_049 | doc_014, doc_015 |
| grep_agent_default | q_078 | medium | 0.50 | doc_040, doc_028 | doc_041, doc_042 |
| grep_agent_default | q_115 | high | 0.00 | doc_011, doc_051, doc_052, doc_053 | doc_013, doc_024, doc_029, doc_030 |
| grep_agent_default | q_117 | medium | 0.50 | doc_007, doc_057 | doc_024, doc_001 |
| adaptive_default | q_023 | high | 0.00 | doc_018 | doc_048, doc_026, doc_049 |
| adaptive_default | q_028 | medium | 0.50 | doc_009, doc_021 | doc_003, doc_036, doc_049 |
| adaptive_default | q_030 | medium | 0.50 | doc_006, doc_019 | doc_036, doc_026, doc_058, doc_009 |
| adaptive_default | q_037 | medium | 0.50 | doc_010, doc_003 | doc_060, doc_048, doc_009, doc_001 |
| adaptive_default | q_046 | high | 0.00 | doc_003, doc_021 | doc_049 |
| adaptive_default | q_084 | medium | 0.50 | doc_037, doc_026 | doc_048, doc_044 |
| adaptive_default | q_085 | medium | 0.50 | doc_036, doc_027 | doc_047, doc_049, doc_048 |
| adaptive_default | q_088 | medium | 0.50 | doc_036, doc_027 | doc_025, doc_049 |
| adaptive_default | q_095 | medium | 0.50 | doc_055, doc_042 | doc_047, doc_028 |
| adaptive_default | q_120 | medium | 0.50 | doc_006, doc_036 | doc_019, doc_049, doc_058 |