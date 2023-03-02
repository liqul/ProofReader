from langchain import PromptTemplate
from langchain.document_loaders import PagedPDFSplitter
import os
from langchain.llms import AzureOpenAI

# todo: replace the path of the paper in pdf 
loader = PagedPDFSplitter(r"/path/to/paper.pdf")
pages = loader.load_and_split()


DEFAULT_REFINE_PROMPT_TMPL = (
    "You are a reviewer of a conference on computer science. You are going to review a paper. \n"
    "To write the review, you first summarize the main content of this paper and then discuss its pros and cons.  \n"
    "The original review comments of the paper is as follows: {review}\n"
    "We have the opportunity to refine the existing review comments."
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new page segment, refine the original review comments to better "
    "assess the paper quality. "
    "If you do update it, please update the sources as well. "
    "If the context isn't useful, return the original answer. "
    "Please start your comments with <START> and end with <END> \n"
    "Your new review comments are: "
)
DEFAULT_REFINE_PROMPT = PromptTemplate(
    input_variables=["review", "context_str"],
    template=DEFAULT_REFINE_PROMPT_TMPL,
)

DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "You are a reviewer of a conference in computer science. You are going to review a paper. \n"
    "To write the review, you first summarize the main content of this paper and then discuss its pros and cons.  \n"
    "Finally, you draw a conclusion about whether to accept or reject the paper. \n"
    "The pages of this paper is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "write a review for this paper. "
    "Please start your comments with <START> and end with <END> \n"
)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    input_variables=["context_str"], template=DEFAULT_TEXT_QA_PROMPT_TMPL
)

# todo: replace the parameters in the environment variables
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://{deployment_name}.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "{openai-key}"

llm = AzureOpenAI(deployment_name="{deployment_name}", model_name="chatgpt", top_p=0.95, max_tokens=1024)

prompt = DEFAULT_TEXT_QA_PROMPT.format(context_str=pages[0])

response = llm(prompt)


first_review = str(response)
reviews = [first_review]

for i in range(1, len(pages)):
    response = str(llm(DEFAULT_REFINE_PROMPT.format(review=reviews[i-1], context_str=pages[i])))
    new_review = response.split("<START>")[1].split("<END>")[0]
    print('-----------------------------------------------')
    print(new_review)
    reviews.append(str(new_review))

# todo: the reviews are stored in reviews array and the last one is supposed to be the best one.




