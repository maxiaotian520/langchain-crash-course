from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequence of input messages.
#   The SystemMessage sets the broad context for the conversation.
# HumanMessagse:
#   Message from a human to the AI model.
# HumanMessage
# A message sent from the perspective of a human to the model. For example, "What is the name of the most populous state in the USA?" or "I love programming".
# AIMessage
# A message sent from the perspective of the AI that the human is interacting with. For example, "The purpose of model regularization is to prevent overfitting 
# in machine learning models".
# SystemMessage
# A message that sets the objectives for the AI to follow. For example, "You're a helpful assistant". SystemMessages are usually passed in as the first message 
# in a sequence of input messages. 

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")


# AIMessage:
#   Message from an AI.
# Answering the last question will be based on it's former questions and so on. 
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")
