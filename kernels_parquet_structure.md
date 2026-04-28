# Parquet File Structure: `kernels.parquet`

This file documents the structure of the `.parquet` files exported by `scripts/export_kernels.py` from the Neon PostgreSQL database.

Each row in the dataset corresponds to a single CUDA kernel discovered and ingested by the pipeline. 

## Schema Columns

| Column | Type | Description |
|---|---|---|
| `id` | Integer | Unique identifier for the kernel record in the database. |
| `repo_name` | String | The GitHub repository name (e.g., `owner/repo`) where the kernel was found. |
| `file_path` | String | The relative path to the file containing the kernel within the repository. |
| `commit_hash` | String | The exact Git commit hash (40 characters) the kernel was scraped from. |
| `raw_code` | String | The full raw source code block of the kernel function. Can be quite large. |
| `code_hash` | String | SHA-256 hash of the `raw_code` to guarantee code uniqueness across the dataset. |
| `domain_tag` | String | High-level categorization or tag for the kernel's domain (e.g., `machine_learning`). |
| `algorithmic_intent` | String | Description of the algorithmic goal or mathematical purpose of the kernel. |
| `memory_pattern` | String | Analyzed memory access patterns (e.g., *row-major, coalesced global memory access*). |
| `hardware_utilization` | String | Estimated hardware utilization characteristics, occupancy bounds, or intensity profile. |
| `mathematical_formulation` | String | The mathematical formulas underlying the computation, if analyzed. |
| `thread_to_data_mapping` | String | Description of how individual CUDA threads or thread blocks map to data elements. |
| `bottleneck_analysis` | String | Potential compute or memory bottlenecks (e.g., compute-bound, branching divergence). |
| `edge_case_vulnerabilities` | String | Potential bugs, numerical instabilities, or unchecked edge cases in the code. |
| `ingested_at` | Timestamp | Timestamp indicating when the kernel was scraped and saved into the database. |

## Usage
The dataset is exported with Snappy compression to balance disk space and read performance. It can be natively loaded into Pandas, Polars, or any machine learning framework supporting PyArrow tables.

```python
import pandas as pd
df = pd.read_parquet("kernels.parquet")
```
