import gradio as gr
from src.vectorstore import load_vectorstore

K = 5

# Load once when the app starts
store = load_vectorstore()


def retrieve(question, k):
    if not question.strip():
        return "Enter a question."

    results = store.similarity_search_with_score(
        question,
        k=int(k)
    )

    output = []

    for i, (doc, score) in enumerate(results, start=1):
        m = doc.metadata

        output.append(
            f"""
### Result {i}

**Score:** `{score:.4f}`  
**Document:** `{m['document_id']}`  
**Page:** `{m['page'] + 1}`  
**Chunk:** `{m['chunk_id']}`

> {doc.page_content}
"""
        )

    return "\n---\n".join(output)


with gr.Blocks(title="ResearchVault Retrieval Lab") as demo:

    gr.Markdown(
        """
        # 🔎 ResearchVault — Retrieval Lab

        Test your vector retrieval and inspect exactly which chunks
        are being returned for each question.
        """
    )

    with gr.Row():

        with gr.Column(scale=2):

            question = gr.Textbox(
                label="Question",
                placeholder="e.g. What is self attention?",
                lines=2
            )

            k = gr.Slider(
                minimum=1,
                maximum=100,
                value=K,
                step=1,
                label="Number of results (Top-K)"
            )

            retrieve_btn = gr.Button(
                "Retrieve",
                variant="primary"
            )

        with gr.Column(scale=3):

            results = gr.Markdown(
                label="Retrieved Chunks"
            )

    retrieve_btn.click(
        fn=retrieve,
        inputs=[question, k],
        outputs=results
    )


if __name__ == "__main__":
    demo.launch()