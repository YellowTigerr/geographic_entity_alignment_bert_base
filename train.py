import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_cosine_schedule_with_warmup
import numpy as np

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义文本相似度数据集类
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



# 定义模型，这里我们不仅计算两段文本的[CLS] token的点积，而是整个句向量的余弦相似度
class BertSimilarityModel(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(BertSimilarityModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = torch.nn.Dropout(p=0.1)  # 引入Dropout层以防止过拟合
        # self.fc1 = torch.nn.Linear(768,128)
        # self.fc2 = torch.nn.Linear(128,2)
        self.fc1 = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask,token_type_ids):

        # 获取bert输出的隐藏层特征
        embeddings = self.dropout(self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)['last_hidden_state'])
        # 把token embedding平均得到sentences_embedding
        sentences_embeddings = torch.mean(embeddings, dim=1)
        sentences_embeddings = sentences_embeddings.squeeze(1)

        # 把sentences_embedding输入分类网络
        # hidden = nn.functional.relu(self.fc1(sentences_embeddings))
        # out = self.fc2(hidden)
        out = self.fc1(sentences_embeddings)
        return out


def train_model(model, train_loader, val_loader, epochs=10, model_save_path='./output/510_1_bert_similarity_model.pth'):
    model.to(device)
    # criterion = SmoothL1Loss()  # 使用自定义的Smooth L1 Loss
    # loss_function = torch.nn.BCEWithLogitsLoss()
    loss_function = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=2e-5)  # 调整初始学习率为5e-5
    num_training_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * num_training_steps,
                                                num_training_steps=num_training_steps)  # 使用带有warmup的余弦退火学习率调度

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            # input_ids2 = batch['input_ids2'].to(device)
            # attention_mask2 = batch['attention_mask2'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            # print(outputs)
            # print(outputs.shape)
            # print(label.shape)
            # loss = criterion(outputs, label.unsqueeze(1))
            loss = loss_function(outputs.view(-1, 2), label.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_loss = 0
            total_val_samples = 0

            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                # input_ids2 = batch['input_ids2'].to(device)
                # attention_mask2 = batch['attention_mask2'].to(device)
                label = batch['label'].to(device)

                val_outputs = model(input_ids, attention_mask,token_type_ids)
                # # val_loss += criterion(val_outputs, label.unsqueeze(1)).item()
                # val_loss += loss_function(val_outputs, label.float()).item()
                val_loss = loss_function(
                    val_outputs.view(-1, 2), label.view(-1))
                total_val_samples += len(label)


            val_loss /= len(val_loader)
            print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)


def collate_to_tensors(batch):
    '''把数据处理为模型可用的数据，不同任务可能需要修改一下，'''
    input_ids = torch.tensor([example['input_ids'] for example in batch])
    attention_mask = torch.tensor([example['attention_mask'] for example in batch])
    token_type_ids = torch.tensor([example['token_type_ids'] for example in batch])
    # input_ids2 = torch.tensor([example['input_ids2'] for example in batch])
    # attention_mask2 = torch.tensor([example['attention_mask2'] for example in batch])
    label = torch.tensor([example['label'] for example in batch])

    return {'input_ids': input_ids, 'attention_mask': attention_mask,  "token_type_ids":token_type_ids,'label': label}


# 加载数据集和预训练模型
tokenizer = BertTokenizer.from_pretrained('./models/chinese-roberta-wwm-ext')
model = BertSimilarityModel('./models/chinese-roberta-wwm-ext')

# 加载数据并创建
train_data = TextSimilarityDataset('./datasets/train.json', tokenizer)
val_data = TextSimilarityDataset('./datasets/val.json', tokenizer)
# test_data = TextSimilarityDataset('../data/STS-B/STS-B.test - 副本.data', tokenizer)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_to_tensors)
val_loader = DataLoader(val_data, batch_size=128, collate_fn=collate_to_tensors)
# test_loader = DataLoader(test_data, batch_size=32, collate_fn=collate_to_tensors)

optimizer = AdamW(model.parameters(), lr=2e-5)

# 开始训练
train_model(model, train_loader, val_loader)

