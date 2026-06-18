from app.graph.nodes import _get_llm
from langchain_core.messages import HumanMessage


def main():
    print("Testing LLM...\n")

    llm = _get_llm(temperature=0.2)
    response = llm.invoke([HumanMessage(content="Say hello in one sentence.")])

    print("\n🧠 MODEL OUTPUT:")
    print(response.content)


if __name__ == "__main__":
    main()