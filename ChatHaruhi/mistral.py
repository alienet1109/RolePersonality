from .BaseLLM import BaseLLM
import torch
import os
from peft import LoraConfig, LoraModel, PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer,AutoTokenizer, AutoModelForCausalLM
# import bitsandbytes, flash_attn
tokenizer_LLaMA = None
model_LLaMA = None

def initialize_Mistral(use_finetuned_model = False,adaper_paths = None):
    global model_LLaMA, tokenizer_LLaMA
    if tokenizer_LLaMA is None:
        tokenizer_LLaMA = AutoTokenizer.from_pretrained("/data/ranyiting/.cache/modelscope/hub/AI-ModelScope/Mistral-7B-Instruct-v0.2", trust_remote_code=True)
    if not use_finetuned_model or not adaper_paths:
        if model_LLaMA is None or isinstance(model_LLaMA, str):
            model_LLaMA = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                torch_dtype=torch.float16,
                device_map="auto"
            )
    else:
        model_LLaMA = load_fintuned_Mistral(
            adapter_to_merge=adaper_paths
            )
    
        
    return model_LLaMA, tokenizer_LLaMA

def LLaMA_tokenizer(text):
    return len(tokenizer_LLaMA.encode(text))

class ChatMistral(BaseLLM):
    def __init__(self, use_finetuned_model = False,adaper_paths = None,model="Mistral"):
        super(ChatMistral, self).__init__()
        
        self.model, self.tokenizer = initialize_Mistral(use_finetuned_model,adaper_paths)
        self.messages = ""

    def initialize_message(self):
        self.messages = "[INST]"

    def ai_message(self, payload):
        self.messages = self.messages + "\n " + payload 

    def system_message(self, payload):
        self.messages = self.messages + "\n " + payload 

    def user_message(self, payload):
        self.messages = self.messages + "\n " + payload 

    def get_response(self):
        if isinstance(self.model, str):
            self.model, self.tokenizer = initialize_Mistral()
        with torch.no_grad():
            encodeds = self.tokenizer.encode(self.messages+"[/INST]", return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(encodeds, max_new_tokens=2000, do_sample=True)
            decoded = self.tokenizer.batch_decode(generated_ids)

        return decoded[0].split("[/INST]")[1]
        
    def print_prompt(self):
        print(self.messages)
        
def load_fintuned_Mistral(
    adapter_to_merge = None,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model. Must after load_tokenizer.
    """
    

    model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                torch_dtype=torch.float16,
                device_map="auto"
            )
    model = init_adapter(model, adapter_to_merge)

    model.requires_grad_(False)
    model.eval()
    for param in model.parameters():
        if param.device.type == "cuda":
            param.data = param.data.to(torch.bfloat16)

    return model


def init_adapter(
    model: "PreTrainedModel", 
    adapter_to_merge = None,
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    adapter_to_resume = None
    
    if not adapter_to_merge:
        
        return model
    for adapter in adapter_to_merge:
        model: "LoraModel" = PeftModel.from_pretrained(
            model, adapter, offload_folder='offload'
        )
        model = model.merge_and_unload()
    if True:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
    return model

if __name__ =="__main__":
    model_LLaMA, tokenizer_LLaMA = initialize_Mistral(use_finetuned_model=True,
                                                      adaper_paths=[
        ])