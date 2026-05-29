# Download/integration notes

This scaffold does not bundle third-party datasets unless their license allows redistribution.

Recommended procedure:

1. Download public data into `data/external/<source_name>/`.
2. Keep the original files unchanged.
3. Add a `LICENSE.txt` or citation note for each source.
4. Write a source-specific ingestion script.
5. Convert to the common schemas in `data/data_dictionary/`.
