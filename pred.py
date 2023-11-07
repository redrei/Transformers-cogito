import torch
from transformers import OpenAIGPTTokenizer
import numpy as np
import model

temp = 0.85
torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
pad = torch.tensor([tokenizer.pad_token_id], requires_grad=False).to("cuda:0")
x_dec = torch.tensor([tokenizer.pad_token_id], requires_grad=False).to("cuda:0")#tokenizer("roses are red\nViolets are blue\nyou are soon wed\nAnd i will be too\n\n",return_tensors="pt")["input_ids"].to("cuda:0").to(int).squeeze(0)#
x_enc = tokenizer("if not for the sun",return_tensors="pt")["input_ids"].to("cuda:0").to(int)
m = torch.jit.load("model.pt").to("cuda:0")
m.eval()
m.training=False
random = not not not False
for i in range(0, 100):
    if i > 0:
        x_dec_o = x_dec.clone()
        x_dec = torch.cat((pad, x_dec)).to(int)
    x_dec = x_dec.unsqueeze(0)
    x_dec = m(x_enc, x_dec)
    x_dec = x_dec.squeeze(0)
    if random and i > 0:
        x_dec = torch.cat((x_dec_o, torch.multinomial(torch.nn.functional.softmax(x_dec[-1]/temp, dim=0).clone().detach(), 1))).to("cuda:0")
    elif i > 0:
        x_dec = torch.cat((x_dec_o, x_dec[-1].clone().detach().argmax().view(1))).to("cuda:0")
    else:
        if random:
            x_dec = (torch.multinomial(torch.nn.functional.softmax(x_dec[-1]/temp, dim=0).clone().detach(), 1)).to("cuda:0")
        else:
            x_dec = (x_dec[-1].clone().detach().to("cpu").argmax().view(1)).to("cuda:0")
print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(x_dec)))