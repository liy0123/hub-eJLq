import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class DeepClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(DeepClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    final_loss = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        final_loss = running_loss / len(dataloader)
    return final_loss


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

output_dim = len(label_to_index)
index_to_label = {i: label for label, i in label_to_index.items()}

test_texts = ["帮我导航到北京", "查询明天北京的天气"]


print("\n【模型1】简单分类器 (2层网络: input → 128 → output)")
print("-" * 70)
simple_model = SimpleClassifier(vocab_size, 128, output_dim)
simple_criterion = nn.CrossEntropyLoss()
simple_optimizer = optim.SGD(simple_model.parameters(), lr=0.01)
simple_final_loss = train_model(simple_model, dataloader, simple_criterion, simple_optimizer, num_epochs=10)
print(f"最终Loss: {simple_final_loss:.4f}")

for text in test_texts:
    pred = classify_text(text, simple_model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"输入 '{text}' → 预测: '{pred}'")

print("\n【模型2】深度分类器 (4层网络: input → 256 → 128 → 64 → output)")
print("-" * 70)
deep_model = DeepClassifier(vocab_size, 256, 128, 64, output_dim)
deep_criterion = nn.CrossEntropyLoss()
deep_optimizer = optim.SGD(deep_model.parameters(), lr=0.01)
deep_final_loss = train_model(deep_model, dataloader, deep_criterion, deep_optimizer, num_epochs=10)
print(f"最终Loss: {deep_final_loss:.4f}")

for text in test_texts:
    pred = classify_text(text, deep_model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"输入 '{text}' → 预测: '{pred}'")

print("\n" + "="*70)
print("对比结果")
print("="*70)
print(f"简单模型最终Loss: {simple_final_loss:.4f}")
print(f"深度模型最终Loss: {deep_final_loss:.4f}")
print(f"\nLoss差异: {abs(simple_final_loss - deep_final_loss):.4f}")
