import math

def recall_at_k(retrieved,relevant,k):
    """Compute recall@k for a single question."""
    
    if not relevant:
        return None   
    else:
        retr=[]
        for paper,page in retrieved[:k]:
                if  (paper,page) in relevant:
                      retr.append((paper,page))             
        return len(set(retr))/len(set(relevant))
   



def reciprocal_rank(retrieved,relevant,k):
    """ compute rr for a single Question"""
    if not relevant:
       return None
    else:    
        for paper,page in retrieved[:k]:
            if (paper,page) in relevant:
                rr=1/(retrieved.index((paper,page))+1)
                return rr 

        else:
            rr=0
            return rr

              
def mean(values):
    try:
       mrr=sum([value for value in values if value is not None])/len([value for value in values if value is not None])
       return mrr

    except ZeroDivisionError:
        return None


def ndcg_at_k(retrieved, relevant, k):
    """nDCG@k with binary relevance at (paper, page) level.
    Returns None when there are no relevant pages."""
    relevant_set = set(relevant)
    if not relevant_set:
        return None

    seen = set()
    dcg = 0.0
    for rank, pair in enumerate(retrieved[:k], start=1):
        if pair in relevant_set and pair not in seen:
            seen.add(pair)
            dcg += 1 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg        