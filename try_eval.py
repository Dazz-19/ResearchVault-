from src.evaluation.dataset import load_test_cases
from src.retrieval import get_retrieval_results
from src.evaluation.metrics import recall_at_k, mean, ndcg_at_k, reciprocal_rank
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.vectorstore import load_vectorstore
from sentence_transformers import CrossEncoder



load_dotenv(override=True)
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0, max_tokens=1024)


# RETRIEVAL_REWRITE_PROMPT = """You rewrite a question into a search query for a
# vector search over research paper chunks. The query is used ONLY for retrieval.

# Rules:
# 1. Remove meta-phrases such as "the X paper", "the authors", "the study".
# 2. Keep every important term from the original question.
# 3. Add a few generic synonyms for the question's intent
#    (e.g. "datasets" -> "training data, corpus"; "optimizer" -> "learning rate").
# 4. Never add specific facts, names, numbers or model names that are not in the question.
# 5. Do not answer the question.
# 6. Output one line of plain text. No quotes, no labels.

# Example 1
# question: What datasets was the Transformer trained on for translation?
# output: Transformer training data corpus datasets translation

# Example 2
# question: Why do the authors call it a chain of thought rather than a solution?
# output: why called chain of thought rather than solution reasoning steps"""


def convert_retrieved_results(results):
    output=[(doc.metadata["document_id"],doc.metadata["page"]+1)for doc,score in results]
    return output 


def convert_test_relevant_questions(relevant_pages):
    con=[(relevant_page["paper"],relevant_page["page"]) for relevant_page in relevant_pages]
    return con 

#not using this function anymore, but keeping it for reference since scored less 
# def rewrite_retrieval_query(question):
#     """Rewrite the question into a search query for the vector store."""
    
#     user_message = question
#     response = llm.invoke(
#         [
#             SystemMessage(content=RETRIEVAL_REWRITE_PROMPT),
#             HumanMessage(content=user_message),
#         ]
#     )
#     rewritten = response.content.strip()
#     return rewritten if rewritten else question




#scored better than BAAI 
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

def retrieve_with_rerank(vectorstore, question: str, k: int = 5, fetch_k: int = 50):
    # embedding search
    candidates = vectorstore.similarity_search(question, k=fetch_k)

    # score each (question, chunk) pair together
    pairs = [(question, d.page_content) for d in candidates]
    scores = reranker.predict(pairs)

    # sort by reranker score, keep the best k
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:k]] 








with open("results_reranker_no_rewriting_new_model.txt", "w") as f:
    vectorstore = load_vectorstore()
    cases=load_test_cases()
    cases_set=[case for case in cases if case["category"]=="answerable"]
    rr_values_10=[]
    dcg_values_10=[]
    recall_values_10=[]
    rr_values_5=[]
    dcg_values_5=[]
    recall_values_5=[]
    for case in cases_set:
        f.write("the question is:\n")
        f.write(f"{case['id']} | {case['question']}\n")
        f.write("relevant pages are:\n")
        rel_pages=case["relevant_pages"]
        converetd_relevant=convert_test_relevant_questions(rel_pages)
        f.write(f"{converetd_relevant}\n")
        f.write("retrieved pages are:\n")
        
        retrieved=retrieve_with_rerank(vectorstore,case["question"], k=10)
        converted_retrieved=convert_retrieved_results(retrieved)
        f.write(f"{converted_retrieved}\n")

        f.write("recall@k for k=10 is:\n")
        recall=recall_at_k(converted_retrieved,converetd_relevant,  k=10)
        recall_values_10.append(recall)
        f.write(f"{recall}\n")
        f.write("NDCG@k for k=10  is:\n")
        dcg=ndcg_at_k(converted_retrieved,converetd_relevant,  k=10)
        dcg_values_10.append(dcg)
        f.write(f"{dcg}\n")
        f.write("reciprocal rank for k=10 is:\n")
        rr=reciprocal_rank(converted_retrieved,converetd_relevant,  k=10)
        rr_values_10.append(rr)
        f.write(f"{rr}\n")


        f.write("recall@k for k=5 is:\n")
        recall=recall_at_k(converted_retrieved,converetd_relevant,  k=5)
        recall_values_5.append(recall)
        f.write(f"{recall}\n")
        f.write("NDCG@k for k=5 is:\n")
        dcg=ndcg_at_k(converted_retrieved,converetd_relevant,  k=5)
        dcg_values_5.append(dcg)
        f.write(f"{dcg}\n")
        f.write("reciprocal rank for k=5 is:\n")
        rr=reciprocal_rank(converted_retrieved,converetd_relevant,  k=5)
        rr_values_5.append(rr)
        f.write(f"{rr}\n")

    f.write("mean reciprocal rank is at k=10:\n")
    f.write(f"{mean(rr_values_10)}\n")
    f.write("mean ndcg is at k=10:\n")
    f.write(f"{mean(dcg_values_10)}\n")   
    f.write("mean recall is at k=10:\n")
    f.write(f"{mean(recall_values_10)}\n") 
    f.write("mean reciprocal rank is at k=5:\n")
    f.write(f"{mean(rr_values_5)}\n")
    f.write("mean ndcg is at k=5:\n")
    f.write(f"{mean(dcg_values_5)}\n")   
    f.write("mean recall is at k=5:\n")
    f.write(f"{mean(recall_values_5)}\n") 







# cases=load_test_cases()
# missing_case=[case for case in cases if case["id"]==11 or case["id"]==31]
# for case in missing_case:
#     rewritten_question = rewrite_retrieval_query(case["question"])
#     print(f"Q{case['id']} | original: {case['question']}")
#     print(f"Q{case['id']} | rewritten: {rewritten_question}")
#     results=get_retrieval_results(rewritten_question, k=20)
#     results=convert_retrieved_results(results)
#     rels=convert_test_relevant_questions(case["relevant_pages"])
#     rr=reciprocal_rank(results,rels,k=20)
#     print(f"Q{case['id']} | RR@20: {rr}")
#     dcg=ndcg_at_k(results,rels,k=20)
#     print(f"Q{case['id']} | nDCG@20: {dcg}")
#     recall=recall_at_k(results,rels,k=20)
#     print(f"Q{case['id']} | Recall@20: {recall}")
#     print("--------------------")
#     print(results)
#     print(rels)
#     print("--------------------")


