from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4")

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create individual runnables (steps in the chain)
# Here each runnable can be viewed as a task
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(response)

## under the hood
RunnableSequence:
.. code-block:: python

    from langchain_core.runnables import RunnableLambda

    def add_one(x: int) -> int:
        return x + 1

    def mul_two(x: int) -> int:
        return x * 2

    runnable_1 = RunnableLambda(add_one)
    runnable_2 = RunnableLambda(mul_two)
    sequence = runnable_1 | runnable_2
    # Or equivalently:
    # sequence = RunnableSequence(first=runnable_1, last=runnable_2)
    sequence.invoke(1)
    await sequence.ainvoke(1)

    sequence.batch([1, 2, 3])
    await sequence.abatch([1, 2, 3])
