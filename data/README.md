# Data

## `uniprot_human_sample.fasta`

2,000 real human protein sequences, used to test batching strategies on realistic, mixed-length data.

| | |
|---|---|
| Source | [UniProtKB/Swiss-Prot](https://www.uniprot.org), reviewed human proteins (organism 9606), downloaded October 2026 |
| Query | `(reviewed:true) AND (organism_id:9606)`: 20,431 proteins |
| Filter | Proteins of at most 1,022 residues, the ESM-2 limit: 18,128 (89%) |
| Sampling | 2,000 drawn at random with `random.Random(0).sample` |
| Lengths | 8 to 1,021 residues, median 365, mean 410, 90th percentile 751 |

Headers are kept as in UniProt (`>sp|accession|name ...`).

## License

UniProt data is available under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

> The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. *Nucleic Acids Research*.
