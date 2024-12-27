from typing import List, Literal
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
import operator
from typing import Annotated, List, Literal, TypedDict
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

# Reference: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/summarization.ipynb

####################
map_template = "Write a concise summary of the following:\\n\\n{context}"

reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes.
"""

map_prompt = ChatPromptTemplate([("system", map_template)])
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
######################


### General Functions
async def _reduce(input: dict, config: RunnableConfig) -> str:

    llm = config["configurable"]["llm"]
    prompt = reduce_prompt.invoke(input)
    response = await llm.ainvoke(prompt)
    return response.content

def length_function(documents: List[Document], config: RunnableConfig) -> int:

    llm = config["configurable"]["llm"]
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)


### Define schema of states:

# Overall state
class OverallState(TypedDict):
    contents: List[str] # list of document splits
    summaries: Annotated[list, operator.add] # each summery corresponds to a content
    collapsed_summaries: List[Document]
    final_summary: str

# Private state
class SummaryState(TypedDict):
    content: str


### Define node and edge functions

# edge: START --> `generate_summary`
def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"] ## each content has in max 1000 tokens??
    ]

# node
async def generate_summary(state: SummaryState, config: RunnableConfig):
    llm = config["configurable"]["llm"]
    prompt = map_prompt.invoke(state["content"])
    response = await llm.ainvoke(prompt)
    return {"summaries": [response.content]}

# next node
def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }


# edge: `collect_summaries`  --> `collapse_summaries` or `generate_final_summary`
#       `collapse_summaries` --> `collapse_summaries` or `generate_final_summary`
def should_collapse(
    state: OverallState,
    config: RunnableConfig
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"], config) # Sum of all the document contents

    token_max = config["configurable"]["token_max"]
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"

# Node
async def collapse_summaries(state: OverallState, config: RunnableConfig):

    # docs: List(Document)
    # returns: List(List(Document))
    doc_lists = split_list_of_docs(docs=state["collapsed_summaries"], # e.g. the sum length of all document contents are (100*100) 10000 ---> 10 * 1000 ---> sum
                                length_func=length_function,
                                token_max=config["configurable"]["token_max"])
    
    # List(List(Document)) ---> List(Document)
    results = [] 
    for doc_list in doc_lists:

        # `acollapse_docs` just merges the metadatas of a set of documents after executing a collapse function on them.
        # List[Document] ---> Document 
        results.append(await acollapse_docs(doc_list, _reduce))    # 1 hour   ############################## ????


    # Explanation!!!
    # 200 (number of summaries) * 100 (average length of each summary) = 20000
    # 20 * (10 * 100 to be near 1000) = 20000
    # 20 * 100 = 2000
    #  2 * (10 * 100) = 2000
    #  2 * 100 = 200 < 1000

    return {"collapsed_summaries": results}


# node
async def generate_final_summary(state: OverallState, config: RunnableConfig):
    response = await _reduce(state["collapsed_summaries"], config)
    return {"final_summary": response}


def load_agent():
    # Construct the graph
    # Nodes
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)  # same as before
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    # Edges
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    app = graph.compile()
    return app