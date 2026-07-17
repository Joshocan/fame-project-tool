# Overall vs Top-k Comparison

| Pipeline | Selection | N | Expected N | Semantic F1 | Coverage |
|---|---|---:|---:|---:|---:|
| SS-RAG | All | 164 |  | 0.6137 | 30.0599 |
| SS-RAG | Top-1 | 1 | 1 | 0.6672 | 39.15 |
| SS-RAG | Top-3 | 3 | 3 | 0.7443 | 46.3533 |
| SS-RAG | Top-5 | 5 | 5 | 0.7397 | 45.884 |
| IS-RAG | All | 80 |  | 0.7154 | 42.8852 |
| IS-RAG | Top-1 | 1 | 1 | 0.7931 | 52.63 |
| IS-RAG | Top-3 | 3 | 3 | 0.7543 | 50.95 |
| IS-RAG | Top-5 | 5 | 5 | 0.7653 | 51.094 |
| SS-NonRAG | All | 161 |  | 0.5689 | 26.2802 |
| SS-NonRAG | Top-1 | 1 | 1 | 0.7024 | 40.26 |
| SS-NonRAG | Top-3 | 3 | 3 | 0.7015 | 40.7567 |
| SS-NonRAG | Top-5 | 5 | 5 | 0.6946 | 39.634 |
| IS-NonRAG | All | 60 |  | 0.6438 | 36.6212 |
| IS-NonRAG | Top-1 | 1 | 1 | 0.8049 | 61.88 |
| IS-NonRAG | Top-3 | 3 | 3 | 0.7861 | 58.3767 |
| IS-NonRAG | Top-5 | 5 | 5 | 0.7943 | 59.374 |
| ALL | All | 465 |  | 0.6196 | 31.8044 |
| ALL | Top-1 | 4 | 4 | 0.7419 | 48.48 |
| ALL | Top-3 | 12 | 12 | 0.7465 | 49.1092 |
| ALL | Top-5 | 20 | 20 | 0.7485 | 48.9965 |
