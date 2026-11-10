from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.retrieval import retrieve_with_rerank
from src.prompts import SYSTEM_PROMPT_TEMPLATE, QUERY_REWRITING_PROMPT

load_dotenv(override=True)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, max_tokens=1024)

NO_EVIDENCE = "I couldn't find enough information in the selected papers to answer this."


def format_history(history, max_turns=4, max_chars=200):
    """Turn the last few chat turns into plain text for the query rewriter."""
    lines = []
    for turn in history[-max_turns:]:
        assistant = turn["assistant"]
        if len(assistant) > max_chars:
            assistant = assistant[:max_chars] + "..."
        lines.append(f"user: {turn['user']}")
        lines.append(f"assistant: {assistant}")
    return "\n".join(lines)


def build_context(results):
    """Build the context string. Each chunk gets a bracketed label the model can cite."""
    blocks = []
    for doc, _score in results:
        m = doc.metadata
        label = f"[Paper: {m['document_id']} | Page: {m['page'] + 1} | Chunk: {m['chunk_id']}]"
        blocks.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(blocks)


def rewrite_question(question, history):
    """Rewrite the latest question so it makes sense on its own."""
    if not history:
        return question

    user_message = (
        f"History:\n{format_history(history)}\n"
        f"latest question: {question}"
    )
    response = llm.invoke(
        [
            SystemMessage(content=QUERY_REWRITING_PROMPT),
            HumanMessage(content=user_message),
        ]
    )
    rewritten = response.content.strip()
    return rewritten if rewritten else question


def generate_from_results(question, results,history=None):
    if history is None:
        history = []

    rewritten_question = rewrite_question(question, history)
    context = build_context(results)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=rewritten_question)
    ])

    return {
        "answer": response.content,
        "rewritten_question": rewritten_question,
        "results": results,
    }





if __name__ == "__main__":

    import gradio as gr
    import html

    def chat_with_researchvault(question, chat_history, conversation_history):

        if not question.strip():
            return chat_history, "", conversation_history

        try:
            rewritten_question = rewrite_question(
                question,
                conversation_history
            )

            results = retrieve_with_rerank(rewritten_question)

            if not results:
                answer = NO_EVIDENCE
                sources_html = "<p>No relevant sources found.</p>"

            else:
                context = build_context(results)

                system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    context=context
                )

                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=rewritten_question)
                ])

                answer = response.content

                
                source_cards = []

                for index, (doc, score) in enumerate(results, start=1):

                    metadata = doc.metadata

                    paper = metadata.get("document_id", "Unknown Paper")
                    paper = str(paper).replace("_", " ").replace("-", " ").title()

                    page = metadata.get("page")
                    page = page + 1 if page is not None else "Unknown"

                    chunk = metadata.get("chunk_id", "Unknown")

                    chunk_display = str(chunk).split("_")[-1]

                    safe_paper = html.escape(paper)
                    safe_page = html.escape(str(page))
                    safe_chunk = html.escape(str(chunk_display))
                    safe_content = html.escape(doc.page_content)

                    source_cards.append(
    f"""
    <details style="
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        background: var(--background-fill-secondary);
    ">
        <summary style="
            cursor: pointer;
            font-weight: 600;
            font-size: 14px;
        ">
            Source {index}: {safe_paper}

            <div style="
                font-size: 12px;
                font-weight: normal;
                opacity: 0.7;
                margin-top: 5px;
            ">
                Page {safe_page} · Chunk {safe_chunk}
            </div>
        </summary>

        <div style="
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border-color-primary);
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-line;
        ">{safe_content}</div>
    </details>
    """
)

                sources_html = "".join(source_cards)


            chat_history = chat_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]

            conversation_history = conversation_history + [
                {
                    "user": question,
                    "assistant": answer
                }
            ]

            return chat_history, sources_html, conversation_history

        except Exception as e:

            error_message = f"Something went wrong: {str(e)}"

            chat_history = chat_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": error_message}
            ]

            return (
                chat_history,
                "<p>Unable to retrieve sources.</p>",
                conversation_history
            )

    def clear_chat():
        return (
            [],
            "<p>Retrieved citations will appear here.</p>",
            []
        )

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="orange",
            secondary_hue="amber"
        ),
        title="ResearchVault"
    ) as demo:

        conversation_history = gr.State([])

        gr.Markdown(
    """
    <h1 align="center">📚 ResearchVault</h1>
    <h3 align="center">AI-Powered Research Paper Assistant</h3>
    <p align="center">
        Explore research. Discover insights. Verify evidence.
    </p>
    """
)

        with gr.Row():

            with gr.Column(scale=3):

                gr.Markdown("<h2 align='center'>💬 Chat</h2>")

                chatbot = gr.Chatbot(
                    label="ResearchVault Assistant",
                    type="messages",
                    height=530
                )

                question_input = gr.Textbox(
                    placeholder="Ask a question about your research papers...",
                    label="Your question",
                    lines=2
                )

                with gr.Row():

                    send_button = gr.Button(
                        "Send",
                        variant="primary"
                    )

                    clear_button = gr.Button("Clear Chat")

            with gr.Column(scale=2):

                gr.Markdown("<h2 align='center'>📑 Sources & Citations</h2>")

                gr.Markdown(
                    "Click a source to view its retrieved passage."
                )

                sources_output = gr.HTML(
                    value="<p>Retrieved citations will appear here.</p>"
                )

        question_input.submit(
            fn=chat_with_researchvault,
            inputs=[
                question_input,
                chatbot,
                conversation_history
            ],
            outputs=[
                chatbot,
                sources_output,
                conversation_history
            ]
        ).then(
            fn=lambda: "",
            outputs=question_input
        )

        send_button.click(
            fn=chat_with_researchvault,
            inputs=[
                question_input,
                chatbot,
                conversation_history
            ],
            outputs=[
                chatbot,
                sources_output,
                conversation_history
            ]
        ).then(
            fn=lambda: "",
            outputs=question_input
        )

        clear_button.click(
            fn=clear_chat,
            inputs=[],
            outputs=[
                chatbot,
                sources_output,
                conversation_history
            ]
        )

    demo.launch()
