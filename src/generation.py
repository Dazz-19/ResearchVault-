from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.retrieval import get_retrieval_results
from src.prompts import SYSTEM_PROMPT_TEMPLATE, QUERY_REWRITING_PROMPT

load_dotenv(override=True)

llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.0, max_tokens=1024)

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


def answer_question(question, history, k=5, filter_paper=None):
    """Rewrite, retrieve, then answer from the retrieved passages only."""
    rewritten_question = rewrite_question(question, history)
    results = get_retrieval_results(
        rewritten_question, k=k, filter_paper=filter_paper
    )

    if not results:
        return {
            "answer": NO_EVIDENCE,
            "rewritten_question": rewritten_question,
            "results": [],
        }

    context = build_context(results)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=rewritten_question),
        ]
    )

    return {
        "answer": response.content,
        "rewritten_question": rewritten_question,
        "results": results,
    }


if __name__ == "__main__":
    history = []
    print("ResearchVault chat. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        result = answer_question(question, history)

        print(f"\n[rewritten] {result['rewritten_question']}")
        print(f"\nAssistant: {result['answer']}\n")
        print("Sources:")
        for doc, score in result["results"]:
            m = doc.metadata
            print(f"  - {m['document_id']} | p.{m['page'] + 1} | {m['chunk_id']} | {score:.3f}")
        print()

        # Store the original question and the answer, never the retrieved chunks
        history.append({"user": question, "assistant": result["answer"]})