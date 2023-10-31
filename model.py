import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from collections import Counter
from typing import List, Tuple
import pandas
from transformers import OpenAIGPTTokenizer
import pickle

uniques = 40072
sequence_length_enc = 25
sequence_length = 150  # for example, predicting 10th word based on previous 9 words


class BeowulfDataset(Dataset):
    def __init__(self, file_path: str, sequence_length: int, sequence_length_enc: int, picklefile=None):
        # Read the text file
        text = pandas.read_csv(file_path)
        text = text[["title", "content"]]
        text = text.dropna()
        # Remove non-alphanumeric characters and tokenize

        # Build vocabulary
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Create input-output pairs using sliding window
        self.sequence_length = sequence_length
        self.pairs = [(x_a[0], x_a[1][:10000]) for x_a in text.to_numpy()]
        if picklefile is None:
            for i in range(len(self.pairs)):
                if len(self.tokenizer(self.pairs[i][0], return_tensors="pt", truncation=True,
                                      max_length=self.sequence_length, padding="max_length")[
                           "input_ids"].squeeze()) == 0 or len(
                        self.tokenizer(self.pairs[i][0], return_tensors="pt", truncation=True,
                                       max_length=self.sequence_length, padding="max_length")[
                            "input_ids"].squeeze()) == 0:
                    self.pairs.pop(i)
                i -= 1
            self.threes = []
            o = 0
            for pair in self.pairs:
                o += 1
                print("\r", o, end="")
                x, y = pair
                x = self.tokenizer(x, return_tensors="pt", truncation=True, max_length=sequence_length_enc,
                                   padding="max_length")["input_ids"].squeeze()
                y = self.tokenizer(y, return_tensors="pt", truncation=False, padding="max_length",
                                   max_length=self.sequence_length)["input_ids"].squeeze()
                if y is None or len(y) == 0:
                    continue
                for i in range(0, len(y)-sequence_length, 70):
                    y_z = y[i:i+sequence_length].clone()
                    z = y_z.clone()
                    y_z = torch.cat((torch.tensor(self.tokenizer.pad_token_id).view(1), y_z[1:]))
                    self.threes.append((x,y_z,z))
            # with open("./threes.pickle", "wb") as f:
            # pickle.dump(self.threes, f)
        else:
            with open(picklefile, "rb") as f:
                self.threes = pickle.load(f)

    def __len__(self):
        return len(self.threes)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.threes[idx]

    def num_unique_tokens(self):
        return len(self.tokenizer)

    def tokens_to_text(self, tokens: torch.Tensor) -> str:
        """
        Convert a list of token IDs back to a string of text.
        """
        if tokens.shape[0] == 1:
            tokens = [tokens.item()]
        else:
            tokens = tokens.tolist()

        txt = [self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(token)) for token in tokens]
        return ' '.join(txt)

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """
        Convert a list of token IDs back to a string of text.
        """
        out = self.tokenizer(text)
        return out



import math
from matplotlib import pyplot as plt


# torch.manual_seed(0)
class self_attention_pure(nn.Module):
    def __init__(self, device: torch.device = torch.device("cuda:0"), p_in: int = 12, mid: int = 512, dims: int = 0, num_heads: int = 4):
        super().__init__()
        self.num_words = p_in
        self.num_dims = p_in
        self.num_heads = num_heads
        if dims != 0:
            self.num_dims = dims
        assert self.num_dims % num_heads == 0

        self.Q = nn.Linear(self.num_dims, self.num_dims).to(device)
        self.K = nn.Linear(self.num_dims, self.num_dims).to(device)
        self.V = nn.Linear(self.num_dims, self.num_dims).to(device)
        self.softmax = nn.Softmax(dim=3)
        self.preoutput_0 = nn.Linear(self.num_dims, mid).to(device)
        self.preoutput_1 = nn.Linear(mid, self.num_dims).to(device)
        self.gelu = nn.GELU()
        self.device = device
        # self.normalize = nn.LayerNorm([p_in, self.num_dims]).to(device)

    def forward(self, x, mask:bool=False):
        x = x.to(self.device)
        id = x
        x_q = self.Q(x).view(x.shape[0], -1, self.num_heads, self.num_dims // self.num_heads)
        x_k = self.K(x).view(x.shape[0], -1, self.num_heads, self.num_dims // self.num_heads)
        x_v = self.V(x).view(x.shape[0], -1, self.num_heads, self.num_dims // self.num_heads)
        x_q = x_q.permute(0, 2, 1, 3)
        x_k = x_k.permute(0, 2, 1, 3)
        x_v = x_v.permute(0, 2, 1, 3)

        x = nn.functional.scaled_dot_product_attention(x_q, x_k, x_v, dropout_p=0.1, is_causal = mask)
        x = x.reshape(x.shape[0], -1, self.num_dims)
        x = self.preoutput_0(x)
        x = self.preoutput_1(x)
        x = x + id
        x = self.gelu(x)

        return x


class self_attention_w_pos_enc(nn.Module):
    def __init__(self, device: torch.device =torch.device("cuda:0"), words:int=12, dims:int=12):
        super().__init__()
        self.num_words = words
        self.num_dims = dims
        self.embedding = nn.Embedding(uniques, self.num_dims).to(device)
        self.pos_encoding = torch.zeros((self.num_words, self.num_dims))

        for i in range(self.num_words):
            for j in range(self.num_dims):
                if j % 2 == 0:
                    self.pos_encoding[i, j] = math.sin(i / math.pow(10000, j / self.num_dims))
                else:
                    self.pos_encoding[i, j] = math.cos(i / math.pow(10000, j - 1 / self.num_dims))
        self.pos_encoding = self.pos_encoding.to(device)

        self.attention = self_attention_pure(device=device, p_in=self.num_dims, dims=self.num_dims, mid=512)
        self.output = nn.Linear(512, 519).to(device)

    def forward(self, x, device: torch.device =torch.device("cuda:0")):
        x = x.to(device)
        x = self.embedding(x)
        x = x + self.pos_encoding
        x = self.attention(x)
        x = self.output(x)

        return x


class self_attention_w_pos_enc_pure(nn.Module):
    def __init__(self, device=torch.device("cuda:0"), num_words:int=12, num_dims:int=12, out:int=512, num_heads:int=4, emb_size:int=uniques):
        super().__init__()
        self.num_words = num_words
        self.num_dims = num_dims
        self.embedding = nn.Embedding(emb_size, self.num_dims).to(device)
        self.device = device
        self.attention = self_attention_pure(device=device, p_in=self.num_words, dims=self.num_dims, mid=out,
                                             num_heads=num_heads)

    def forward(self, x, device: torch.device =torch.device("cuda:0"), mask:bool=False):

        pos_encoding = torch.zeros((x.shape[1], self.num_dims))
        for i in range(x.shape[1]):
            for j in range(self.num_dims):
                if j % 2 == 0:
                    pos_encoding[i, j] = math.sin(i / math.pow(10000, j / self.num_dims))
                else:
                    pos_encoding[i, j] = math.cos(i / math.pow(10000, (j - 1) / self.num_dims))
        pos_encoding = pos_encoding.to(device)

        x = x.to(self.device)
        x = self.embedding(x)
        x = x+ pos_encoding
        x = self.attention(x, mask=mask)
        return x


class multi_layer_self_attention(nn.Module):
    def __init__(self, device: torch.device =torch.device("cuda:0"), outs:list[int]=[144, 256, 512, 519], seq_length:int=12, dims:int=12, num_heads:int=4):
        super().__init__()
        self.attention_w_enc = self_attention_w_pos_enc_pure(device=device, num_words=seq_length, num_dims=dims,
                                                             out=outs[0], num_heads=num_heads)
        self.attention_1 = self_attention_pure(device=device, p_in=seq_length, dims=dims, mid=outs[0],
                                               num_heads=num_heads)
        self.attention_2 = self_attention_pure(device=device, p_in=seq_length, dims=dims, mid=outs[1],
                                               num_heads=num_heads)
        self.attention_3 = self_attention_pure(device=device, p_in=seq_length, dims=dims, mid=outs[2],
                                               num_heads=num_heads)
        self.linear_1 = nn.Linear(seq_length * dims, outs[0]).to(device)
        self.linear_2 = nn.Linear(outs[0], outs[1]).to(device)
        self.flatten = nn.Flatten(start_dim=1)
        self.output = nn.Linear(outs[1], outs[3]).to(device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.attention_w_enc(x)
        x = self.attention_1(x)
        x = self.attention_2(x)
        x = self.attention_3(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.output(x)
        return x



class self_attention_pure_encoder_qk_decoder_v(nn.Module):
    def __init__(self, device: torch.device =torch.device("cuda:0"), p_in: int=12, mid:int =512, dims:int=0, num_heads:int=4):
        super().__init__()
        self.num_words = p_in
        self.num_dims = p_in
        self.num_heads = num_heads
        if dims != 0:
            self.num_dims = dims
        assert self.num_dims % num_heads == 0

        self.Q = nn.Linear(self.num_dims, self.num_dims).to(device)
        self.K = nn.Linear(self.num_dims, self.num_dims).to(device)
        self.V = nn.Linear(self.num_dims, self.num_dims).to(device)
        self.softmax = nn.Softmax(dim=3)
        self.preoutput_0 = nn.Linear(self.num_dims, mid).to(device)
        self.preoutput_1 = nn.Linear(mid, self.num_dims).to(device)
        # self.normalize = nn.LayerNorm([self.num_dims]).to(device)
        self.gelu = nn.GELU()
        self.device = device

    def forward(self, x, keyvalue):
        x = x.to(self.device)
        keyvalue = keyvalue.to(self.device)
        id = x
        x_q = self.Q(x).view(x.shape[0], -1, self.num_heads, self.num_dims // self.num_heads)
        x_k = self.K(keyvalue).view(x.shape[0], -1, self.num_heads, self.num_dims // self.num_heads)
        x_v = self.V(keyvalue).view(x.shape[0], -1, self.num_heads, self.num_dims // self.num_heads)

        x_q = x_q.permute(0, 2, 1, 3)
        x_k = x_k.permute(0, 2, 1, 3)
        x_v = x_v.permute(0, 2, 1, 3)

        x = nn.functional.scaled_dot_product_attention(x_q, x_k, x_v, dropout_p=0.1, is_causal = False)
        x = x.reshape(x.shape[0], -1, self.num_dims)
        x = self.preoutput_0(x)
        x = self.preoutput_1(x)
        x = x + id
        x = self.gelu(x)

        return x


class multi_layer_self_attention_encoder_decoder_scheme(nn.Module):
    def __init__(self, device: torch.device =torch.device("cuda:0"), outs: list[int]=[144, 256, 512, 519], seq_length: int=12, dims:int=12, num_heads:int=4,
                 emb_size:int=uniques, dropout:float=0.2):
        super().__init__()
        self.attention_w_enc = self_attention_w_pos_enc_pure(device=device, num_words=seq_length, num_dims=dims,
                                                             out=outs[0], num_heads=num_heads, emb_size=emb_size)
        self.attention_0 = self_attention_pure(device=device, p_in=seq_length, dims=dims, mid=outs[2],
                                               num_heads=num_heads)

        self.attention_1 = self_attention_pure_encoder_qk_decoder_v(device=device, p_in=seq_length, dims=dims,
                                                                    mid=outs[0], num_heads=num_heads)
        self.attention_2 = self_attention_pure_encoder_qk_decoder_v(device=device, p_in=seq_length, dims=dims,
                                                                    mid=outs[1], num_heads=num_heads)

        self.pad_token = uniques

        self.attention_w_enc_encoder = self_attention_w_pos_enc_pure(device=device, num_words=sequence_length_enc,
                                                                     num_dims=dims, out=outs[0], num_heads=num_heads,
                                                                     emb_size=emb_size)
        self.attention_1_encoder = self_attention_pure(device=device, p_in=sequence_length_enc, dims=dims, mid=outs[0],
                                                       num_heads=num_heads)
        self.attention_2_encoder = self_attention_pure(device=device, p_in=sequence_length_enc, dims=dims, mid=outs[1],
                                                       num_heads=num_heads)
        self.dims = dims
        self.linear_1 = nn.Linear(dims, outs[0]).to(device)
        self.linear_2 = nn.Linear(outs[0], outs[1]).to(device)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(outs[1], outs[3]).to(device)
        self.device = device


    def forward(self, x_enc, x_dec):
        # Gjør først venstre side av figure 1 for å få query og key
        x_enc = x_enc.to(self.device)
        x_enc = self.attention_w_enc_encoder(x_enc)
        self.dropout(x_enc)
        x_enc = self.attention_1_encoder(x_enc)
        self.dropout(x_enc)
        x_enc = self.attention_2_encoder(x_enc)
        # self.dropout(x_enc)
        # x_enc = self.attention_3_encoder(x_enc, visualization=visualization)

        x_dec = x_dec.to(self.device)

        at_mask = self.training
        x_dec = self.attention_w_enc(x_dec, mask=at_mask)
        x_dec = self.attention_0(x_dec, mask=at_mask)
        self.dropout(x_dec)
        x_dec = self.attention_1(x_dec, x_enc)
        self.dropout(x_dec)
        x_dec = self.attention_2(x_dec, x_enc,)
        self.dropout(x_dec)

        x_dec = self.linear_1(x_dec)
        x_dec = self.linear_2(x_dec)
        x_dec = self.output(x_dec)

        return x_dec


if __name__ == "__main__":
    self_atten = multi_layer_self_attention_encoder_decoder_scheme(seq_length=10, dims=256, num_heads=8)
    sa = torch.jit.script(self_atten)

    file_path = 'gutenberg-poetry-dataset.csv'
    batch_size = 64
    shuffle = True
    sequence_length = 150  # for example, predicting 10th word based on previous 9 words
    sequence_length_enc = 25
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)

    dataset = BeowulfDataset(file_path, sequence_length, sequence_length_enc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Example usage:
    print(f"Number of unique tokens in the dataset: {dataset.num_unique_tokens()}")

    print(dataset[128])
    uniques = dataset.num_unique_tokens()
    print(uniques)

    seq_length = sequence_length
    self_atten = multi_layer_self_attention_encoder_decoder_scheme(seq_length=seq_length, dims=256, num_heads=8,
                                                                   outs=[256, 128, 128, uniques], emb_size=uniques,
                                                                   dropout=0.4)
    # p = self_atten(numdata[:12])

    optimizer = torch.optim.Adam(self_atten.parameters(), lr=0.001, weight_decay=0.005)
    loss = nn.CrossEntropyLoss()
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, cooldown=3, factor=0.2)
    sa = torch.jit.script(self_atten)

    for k in range(50):
        loss_n = 0
        acc = 0
        best = 1000
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = batch[0].view((-1, sequence_length_enc)), batch[1].view((-1, sequence_length))
            z = batch[2]
            z = z.to("cuda:0")

            y_p = self_atten(x, y)
            l = loss(y_p.permute(0,2,1), z)
            l.backward()
            optimizer.step()
            loss_n += l
            acc += torch.eq(z, y_p.argmax(dim=2)).sum().item() / (batch[0].shape[0]*sequence_length)
            print(f"\rit: {i + 1}/{len(dataloader)}, loss: {loss_n / (i + 1)}, acc: {acc / (i + 1)}", end="")
        loss_n /= len(dataloader)
        if loss_n < best:
            best = loss_n
            sa = torch.jit.script(self_atten)
            torch.jit.save(sa, "model.pt")
        acc /= len(dataloader)
        schedule.step(loss_n)
        print("epoch:", k, "loss:", loss_n, "acc:", acc)

    # p = self_atten(numdata[:12])

    # print(p, words[p.argmax()], data[13])

    accuracy = 0
    for i, batch in enumerate(dataloader):
        x, y = batch[0].view((-1, sequence_length_enc)), batch[1].view((-1, sequence_length))
        z = torch.zeros((-1, sequence_length, uniques))
        z[:, :, batch[2].view((-1, sequence_length, 1))[:, :]] = 1
        y_p = self_atten(x, y)
        y_p = y_p.to("cpu")
        # print(y, words[y_p.argmax()])
        accuracy += torch.eq(z.argmax(dim=1), y_p.argmax(dim=1)).sum().item() / batch[0].shape[0]

    accuracy /= len(dataloader) - seq_length - 1
    print("accuracy:", accuracy)

    print(dataset.tokens_to_text(dataset[128][0]), dataset.tokens_to_text(torch.Tensor([dataset[128][1]])),
          dataset.tokens_to_text(torch.Tensor([dataset[128][2]])))
    print(dataset[128][0].shape)
    x = torch.Tensor(dataset[128][0]).unsqueeze(0)
    y = torch.Tensor(dataset[128][1]).unsqueeze(0)

    self_atten(x, y)
    # x = torch.Tensor(dataset[129][0]).unsqueeze(0)
    # self_atten(x, visualization=True)
