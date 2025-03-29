from data_acquisition import download_medical_articles
from preprocessing import MedicalPaperPreprocessor
from embedding import Embedding, get_similarity, graph_embeddings


def main():
    download_medical_articles()
    preprocessor = MedicalPaperPreprocessor(pdf_dir="papers", chunk_size_tokens=500, overlap_tokens=50)
    all_chunks = preprocessor.process_all_pdfs()
    preprocessor.save_chunks_to_json("pdf_chunks.json", all_chunks)

    print(f"Processed {len(preprocessor.pdf_files)} PDFs and saved {len(all_chunks)} chunks to 'pdf_chunks.json'.")
    
    embedder = Embedding()
    embedding = embedder.create_embeddings(all_chunks)
    
    #examples
    print_example(0, 1, embedding, all_chunks)
    print_example(0, 5, embedding, all_chunks)
    print_example(1, 5, embedding, all_chunks)
    
    graph_embeddings(embedding)
    

def print_example(index1, index2, embedding, all_chunks, max_length=40):
    similarity_score = get_similarity(embedding[index1], embedding[index2])
    chunk1_text = all_chunks[index1]["text"]
    chunk2_text = all_chunks[index2]["text"]
    print("-------------------------------------------------")
    print("First chunk:\n\'{}\'".format(chunk1_text[:max_length] + ("..." if len(chunk1_text) > max_length else "")))
    print("-------------------------------------------------")
    print("Second chunk:\n\'{}\'".format(chunk2_text[:max_length] + ("..." if len(chunk2_text) > max_length else "")))
    print("-------------------------------------------------")
    print("Similarity score: {}".format(similarity_score))
    print("-------------------------------------------------\n\n")

if __name__ == "__main__":
    main()
