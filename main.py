from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

llm = OpenAI()

# first prompt template
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)
# second prompt template
test_prompt = PromptTemplate(
    template="Write a test for the following {language} code:\n{code}",
    input_variables=["language", "code"]
)

# first chain
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

# second chain
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

#  wire chains together in series
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["test", "code"]
)

result = chain({
    "language": args.language,
    "task": args.task

})

# print(result)
print(">>>>>> GENERATED CODE:")
print(result["code"])

print(">>>>>> GENERATED TEST:")
print(result["test"])