from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time


class Brain:

    def __init__(self):

        self.model_name = "microsoft/phi-3-mini-4k-instruct"

        print("=" * 60)
        print("Loading Model.....")
        print("=" * 60)

        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        print("[✓] Tokenizer Loaded")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        end = time.time()

        print("[✓] Model Loaded Successfully")
        print(f"[INFO] Loading Time : {end-start:.2f} seconds")

        print("[INFO] Torch Version :", torch.__version__)
        print("[INFO] CUDA Available :", torch.cuda.is_available())
        print("[INFO] CUDA Device Count :", torch.cuda.device_count())

        if torch.cuda.is_available():
            print("[INFO] GPU :", torch.cuda.get_device_name(0))

        print("[INFO] Model Device :", next(self.model.parameters()).device)

        print("=" * 60)
        print("Brain Loaded !")
        print("=" * 60)

    def think(self, prompt):

        print("\n================= DEBUG =================")

        try:

            print("[1] User Prompt Received")
            print(prompt)

            print("\n[2] Tokenizing Input...")

            token_start = time.time()

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            )

            token_end = time.time()

            print(
                f"[✓] Tokenization Completed ({token_end-token_start:.3f} sec)"
            )

            print("\n[3] Moving Inputs To Model Device...")

            inputs = inputs.to(self.model.device)

            print("[✓] Inputs moved to :", self.model.device)

            print("\n[4] Starting Generation...")

            gen_start = time.time()

            with torch.no_grad():

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7
                )

            gen_end = time.time()

            print(
                f"[✓] Generation Completed ({gen_end-gen_start:.2f} sec)"
            )

            print("\n[5] Decoding Output...")

            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            print("[✓] Decode Completed")

            print("================ DEBUG END ================\n")

            return response

        except Exception as e:

            print("\n************ ERROR ************")
            print(type(e).__name__)
            print(e)
            print("*******************************")

            raise