import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast


class LanguageModel:
    """
    A class that uses a pretrained language model to generate paraphrases and answers.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (AutoModelForCausalLM): The language model used for generating paraphrases and answers.
    """

    def __init__(self, model_name):
        """
        Initializes the language model and tokenizer.

        Args:
            model_name (str): The name of the model to use (e.g., 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    def generate_paraphrases(self, query, n=3):
        """
        Generates a list of paraphrases for a given query.

        Args:
            query (str): The input query for which paraphrases will be generated.
            n (int): The number of paraphrases to generate. Defaults to 3.

        Returns:
            list: A list of generated paraphrases.
        """
        prompt = f"Generate {n} paraphrases for the following query: {query}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, do_sample=True, max_new_tokens=128)
        paraphrases = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return paraphrases

    def generate_answer(self, question, context):
        """
        Generates an answer for a given question based on the provided context.

        Args:
            question (str): The question to be answered.
            context (str): The context or document to answer the question from.

        Returns:
            str: The generated answer.
        """
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with autocast():
            outputs = self.model.generate(**inputs, do_sample=True, max_new_tokens=128)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
