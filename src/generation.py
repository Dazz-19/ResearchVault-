from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.retrieval import retrieve_with_rerank
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


