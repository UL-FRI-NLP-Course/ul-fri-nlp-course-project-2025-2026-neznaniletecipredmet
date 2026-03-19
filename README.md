# Natural language processing course: `Chatbot for UL FRI students`

~Please, organize README and the whole structure of the repository to be self-contained and reproducible.~


## Dataset

We will be using publicly available data from the official Fakulteta za računalništvo in informatiko and Univerza v Ljubljani websites. 
Links to the raw data used are available in repository folder **raw_dataset**.

## Initial ideas

We currently plan to use Retrieval-augmented generation (RAG) to fine-tune the LLM on our data. The data retrieval will likely be performed by first scraping the web pages and downloading relevant PDFs, then splitting the data into sensible chunks, e.g. by paragraphs.

Some of the libraries and technologies we might use during development:
- langchain (for the AI agent framework)
- PyMuPDF (to extract text from PDFs)
- pytesseract (if extracting text directly from PDF documents fails, we will need to perform optical character recognition)