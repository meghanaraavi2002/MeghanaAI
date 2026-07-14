import torch
import torch.nn as nn


class Tokenizer():
    
    def __init__(self,value):
        self.value=value
        self.char = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.,!?-+=_@#$%^&*()[]{};:'\"<>?/\\"
        self.vocSize=len(self.char)
        self.int_to_char={i:char for i,char in enumerate(self.char) }
        self.char_to_int={char:i for i, char in enumerate(self.char)}
    def encode(self):
        retVal=[]
        for i in self.value:
            if i not in self.char:
                k=list(self.int_to_char.keys())[-1]+1
                self.int_to_char[k]=i
                self.char+=i
                self.char_to_int[i]=k
                self.vocSize+=1
            retVal.append(self.char_to_int[i])
        return retVal
    def decode(self,integer_list):
        retVal=[]
        for i in integer_list:
            retVal.append(self.int_to_char[i])
        return retVal
t=Tokenizer("Jarvis")
encodes=t.encode()
tensorid=torch.tensor(encodes,dtype=torch.long)
embd_layer=nn.Embedding(t.vocSize,4)
embd_vector=embd_layer(tensorid)
pos_tensor=torch.randn([6,4])
fianl_output=embd_vector+pos_tensor
queries_layer=nn.Linear(4,4,bias=False)
keys_layer=nn.Linear(4,4,bias=False)
values_layer=nn.Linear(4,4,bias=False)
Q=queries_layer(fianl_output)
K=keys_layer(fianl_output)
V=values_layer(fianl_output)
scores=Q@K.T
scaled_scores=scores/2.0
attention_weigh=torch.softmax(scaled_scores,dim=-1)
attention_op=attention_weigh@V
print(attention_op)
print(attention_op.shape)