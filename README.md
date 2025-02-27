# e2p2: End-to-end PDF parsing

- Create an end-to-end PDF parsing pipeline, which can extract text and tables from scanned or digital PDFs into structured documents (markdown, html, word, etc.)
- Manage scanned or digital PDFs
- Multilingual support
- Web app

## References

- [PdfTable: A Unified Toolkit for Deep Learning-Based Table Extraction](https://arxiv.org/abs/2409.05125), code: https://github.com/CycloneBoy/pdf_table?tab=readme-ov-file
- [Table Transformer (TATR) is a deep learning model for extracting tables from unstructured documents ](https://github.com/microsoft/table-transformer?tab=readme-ov-file)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) multilingual OCR toolkits based on PaddlePaddle (practical ultra lightweight OCR system, support 80+ languages recognition, provide data annotation and synthesis tools, support training and deployment among server, mobile, embedded and IoT devices)
- [Nougat: Neural Optical Understanding for Academic Documents](https://github.com/facebookresearch/nougat?tab=readme-ov-file)
- [PdfPlumber](https://github.com/jsvine/pdfplumber) Plumb a PDF for detailed information about each char, rectangle, line, et cetera — and easily extract text and tables.
- [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) adds an OCR text layer to scanned PDF files, allowing them to be searched

- [EasyOCR](https://github.com/JaidedAI/EasyOCR?tab=readme-ov-file)
- [PDF Extract Kit](https://github.com/opendatalab/PDF-Extract-Kit?tab=readme-ov-file)
- [MinerU](https://github.com/opendatalab/MinerU)
- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO?tab=readme-ov-file)
- [Pix2Text](https://github.com/breezedeus/Pix2Text/tree/main)
- [UniMERNet](https://github.com/opendatalab/UniMERNet)


## TODO

- [] add language detection and use it for ocr
- [] add option to ocr model to only detect text but not recognize
- [] add table extraction model with rapid table
- [] add reading order model
- [] create postprocessing module to read through the detections and recreate the output in markdown
- [] add OCR with EasyOCR
- [] TESTS !
