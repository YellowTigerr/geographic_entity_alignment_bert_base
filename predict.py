import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextSimilarityDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.data = pd.read_json(file_path, lines=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        sentence = self.tokenizer.encode_plus(sample['sentence1'], sample['sentence2'], padding='max_length', truncation=True, max_length=self.max_len)
        label = 1 if sample['label']=="exact_match" else 0
        input_ids = sentence['input_ids']
        token_type_ids = sentence["token_type_ids"]
        attention_mask = sentence['attention_mask']
        return {'input_ids':input_ids,
                'token_type_ids':token_type_ids,
                'attention_mask':attention_mask,
                'label':label}

class BertSimilarityModel(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(BertSimilarityModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = torch.nn.Dropout(p=0.1)  # 引入Dropout层以防止过拟合
        self.fc1 = torch.nn.Linear(768,128)
        self.fc2 = torch.nn.Linear(128,2)
        # self.fc1 = torch.nn.Linear(768,2)

    def forward(self, input_ids, attention_mask,token_type_ids):

        # 获取bert输出的隐藏层特征
        embeddings = self.dropout(self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)['last_hidden_state'])
        # 把token embedding平均得到sentences_embedding
        sentences_embeddings = torch.mean(embeddings, dim=1)
        sentences_embeddings = sentences_embeddings.squeeze(1)

        # 把sentences_embedding输入分类网络
        hidden = nn.functional.relu(self.fc1(sentences_embeddings))
        out = self.fc2(hidden)
        # out = self.fc1(sentences_embeddings)
        return out

def collate_to_tensors(batch):
    '''把数据处理为模型可用的数据，不同任务可能需要修改一下，'''
    input_ids = torch.tensor([example['input_ids'] for example in batch])
    attention_mask = torch.tensor([example['attention_mask'] for example in batch])
    token_type_ids = torch.tensor([example['token_type_ids'] for example in batch])
    # input_ids2 = torch.tensor([example['input_ids2'] for example in batch])
    # attention_mask2 = torch.tensor([example['attention_mask2'] for example in batch])
    label = torch.tensor([example['label'] for example in batch])

    return {'input_ids': input_ids, 'attention_mask': attention_mask,  "token_type_ids":token_type_ids,'label': label}


# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('./models/chinese-roberta-wwm-ext')
model = BertSimilarityModel('./models/chinese-roberta-wwm-ext')
model.load_state_dict(torch.load('./output/510_bert_similarity_model.pth'))  # 请确保路径正确
model.eval()  # 设置模型为评估模式
model.to(device)

val_data = TextSimilarityDataset(r'./datasets/test513.json', tokenizer)
test_loader = DataLoader(val_data, batch_size=128, collate_fn=collate_to_tensors)

correct = 0     #预测正确
label_1 = 0
label_0 = 0
pre_1 = 0
pre_0 = 0
tp = 0

# 计算相似度
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        label = batch['label'].to(device)
        test_outputs = model(input_ids, attention_mask,token_type_ids)
        # print(test_outputs)
        softmax = nn.Softmax(dim=1)
        probabilities = softmax(test_outputs)
        predicted_class = torch.argmax(probabilities,dim=1)
        # print("Predicted class:", predicted_class)

        pre_num_zeros = torch.sum(torch.eq(predicted_class, 0)).item()
        pre_num_ones = torch.sum(torch.eq(predicted_class, 1)).item()
        label_num_zeros = torch.sum(torch.eq(label, 0)).item()
        label_num_ones = torch.sum(torch.eq(label, 1)).item()
        correct_predictions = torch.eq(predicted_class, label)
        num_correct = torch.sum(correct_predictions).item()

        num_both_ones = torch.sum((predicted_class == 1) & (label == 1)).item()
        correct+=num_correct
        pre_0+=pre_num_zeros
        pre_1+=pre_num_ones
        label_0+=label_num_zeros
        label_1+=label_num_ones
        tp+=num_both_ones


print("正确的个数:",correct)
print("预测为0:",pre_0)
print("预测为1:",pre_1)
print("标签为0:",label_0)
print("标签为1:",label_1)
print("tp:",tp)

print("准确度:", correct/len(val_data))
print("精确率:",tp / pre_1)
print("召回率:",tp / label_1)

