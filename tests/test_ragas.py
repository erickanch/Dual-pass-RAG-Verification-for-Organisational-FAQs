import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
load_dotenv()

# LangchainLLMWrapper / LangchainEmbeddingsWrapper are deprecated in ragas, but
# the modern llm_factory path is OpenAI-centric. We need an Anthropic judge
# (protocol: judge vendor must differ from both Node A and Node B), so the
# wrappers stay. Ragas is pinned at 0.3.9.

judge = LangchainLLMWrapper(ChatAnthropic(model='claude-sonnet-4-5-20250929', temperature=0))
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))

d = Dataset.from_dict({
    'question': ['What is the capital of France?'],
    'answer': ['The capital of France is Paris.'],
    'contexts': [['France is a country in Europe. Its capital is Paris.']],
})
print(evaluate(d, metrics=[faithfulness, answer_relevancy], llm=judge, embeddings=emb))
