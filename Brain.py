# from transformers import AutoModelForCausalLM, AutoTokenizer
# model_name="microsoft/phi-3-mini-4k-instruct"
# token=AutoTokenizer.from_pretrained(model_name)
# model=AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype="auto")
# prompt="Hello Vennela, how are you ?"
# inputs=token(prompt,return_tensors="pt")
# output=model.generate(**inputs,max_length=100)
# print(token.decode(output[0]))

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configure LoRA
for name, module in model.named_modules():
    print(name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv_proj","o_proj"],  # ✅ matches your model
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
