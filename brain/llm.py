from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Brain:
    def __init__(self):
        self.model_name="microsoft/phi-3-mini-4k-instruct"
        print("Loading Model.....")
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name)
        self.model=AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Brain Loaded !")
    def think(self,prompt):
        input=self.tokenizer(prompt,return_tensors="pt").to(self.model.device)

        outputs=self.model.generate(
            **input,max_new_tokens=200
        )
        return self.tokenizer.decode(outputs[0],skip_special_tokens=True)