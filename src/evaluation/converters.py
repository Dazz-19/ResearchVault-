def convert_retrieved_results(results):
    output=[(doc.metadata["document_id"],doc.metadata["page"]+1)for doc,score in results]
    return output 


def convert_test_relevant_questions(relevant_pages):
    con=[(relevant_page["paper"],relevant_page["page"]) for relevant_page in relevant_pages]
    return con 