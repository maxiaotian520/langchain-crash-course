from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}") # \n{x} 是一个格式化字符串的一部分，用于在输出结果中插入换行符，并将变量 x 的值添加到新行上。

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# Run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(result)

### Results:
Word count: 48
SURE, HERE ARE THREE LAWYER JOKES FOR YOU:

1. WHY DON'T SHARKS ATTACK LAWYERS?
   BECAUSE OF PROFESSIONAL COURTESY!

2. WHAT'S THE DIFFERENCE BETWEEN A LAWYER AND A HERD OF BUFFALO?
   THE LAWYER CHARGES MORE.

3. HOW MANY LAWYER JOKES ARE THERE?
   ONLY THREE. THE REST ARE TRUE STORIES.
