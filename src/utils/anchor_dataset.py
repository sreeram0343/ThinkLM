import json

class AnchorDataset:

    def __init__(self,path):

        with open(path,"r",encoding="utf-8") as f:

            self.data=json.load(f)

    def __len__(self):

        return len(self.data)

    def get(self,index):

        return self.data[index]

    def batch(self,batch_size):

        for i in range(0,len(self.data),batch_size):

            yield self.data[i:i+batch_size]

    def format_prompt(self,item):

        return f"""
Question:
{item['question']}

Context:
{item['context']}

Answer:
"""