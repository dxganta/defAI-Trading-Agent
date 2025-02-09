import os
from IPython.display import Image
from langgraph.graph import StateGraph


def save_graph(graph : StateGraph):
    try:
        graph_image = Image(graph.get_graph().draw_mermaid_png())

        with open("graph.png", "wb") as f:
            f.write(graph_image.data)
        
        print(f"Graph saved successfully to: {os.path.abspath('graph.png')}")
    except Exception as e:
        print(f"Failed to save graph: {str(e)}")
            # This requires some extra dependencies and is optional
        pass