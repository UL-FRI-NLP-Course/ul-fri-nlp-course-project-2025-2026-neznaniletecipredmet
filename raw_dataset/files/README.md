# Manually added files (PDFs, images)

Put small hand-curated files here when they are not fetched via `data_links.txt`.
This folder is parsed by `code/scripts/build_index.py`.

## PDFs
- Place PDFs anywhere under this folder, e.g. `raw_dataset/files/pdfs/your_doc.pdf`.

## Images (campus map, layouts)
This project does not run OCR on standalone image files.

Note: scanned PDFs can still be OCR'd during indexing if PDF parsing uses Docling
and a working OCR backend is available.

- Put images here, e.g. `raw_dataset/files/images/campus_map.png`
- Add a sidecar text file with the same name:
  - `raw_dataset/files/images/campus_map.txt`

The sidecar text is what will be indexed for RAG. In the sidecar, describe the image in a structured way.
Preferably in Slovenian and English combined.