from data_acquisition import download_medical_articles
from preprocessing import MedicalPaperPreprocessor

def main():
    download_medical_articles()
    preprocessor = MedicalPaperPreprocessor(pdf_dir="papers", chunk_size_tokens=500, overlap_tokens=50)
    all_chunks = preprocessor.process_all_pdfs()
    preprocessor.save_chunks_to_json("pdf_chunks.json", all_chunks)

    print(f"Processed {len(preprocessor.pdf_files)} PDFs and saved {len(all_chunks)} chunks to 'pdf_chunks.json'.")

if __name__ == "__main__":
    main()
