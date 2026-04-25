import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import tqdm
import json
import os

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------
# 1) Dataset
# -----------------------------------------------------------------------

class CustomDataset(Dataset):
    """
    In multi-label settings, self.labels will be multi-hot vectors
    of shape [num_classes].
    """
    def __init__(self, embeddings, labels, ranks):
        # Convert labels to float tensor for multi-label BCE loss
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.ranks = ranks

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.ranks[idx]


from sentence_transformers import SentenceTransformer, models

def get_embedding(questions, model_path):
    default_model = models.Transformer('microsoft/codebert-base')
    pooling_model = models.Pooling(default_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[default_model, pooling_model])
    model = SentenceTransformer(model_path)
    embeddings = []
    print('Generating embeddings...')
    for question in tqdm.tqdm(questions, ncols=75):
        embedding = model.encode(question)
        embeddings.append(embedding)
    return embeddings

# -----------------------------------------------------------------------
# 2) Model
# -----------------------------------------------------------------------

class MultiLabelClassificationModel(nn.Module):
    """
    Multi-label version:
      - Output dimension = num_classes
      - Typically we do not apply Sigmoid here but rather return raw logits.
        We will apply BCEWithLogitsLoss in the training step, which internally
        uses a sigmoid.
    """
    def __init__(self, input_size, num_classes):
        super(MultiLabelClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # raw logits (no softmax!)
        return x  # shape: (batch_size, num_classes)


# -----------------------------------------------------------------------
# 3) Loading data and building multi-label vectors
# -----------------------------------------------------------------------

def generate_list(data):
    questions, labels_list, ranks = [], [], []

    for per_data in data:
        # ranks is still stored if needed for nDCG
        # But for multi-label classification, we may adapt your usage of "ranks."
        rank_dict = {str(i):0 for i in range(num_classes)}  # keep your old structure

        question = per_data['prompt']

        # Build a multi-hot vector
        # "ranked_techniques" is a list of techniques. Mark them 1 in the label vector.
        label_vector = np.zeros(num_classes, dtype=np.float32)
        for idx, technique_info in enumerate(per_data['ranked_techniques']):
            technique_name = technique_info[0]
            score = technique_info[1]
            # mark it as relevant
            label_index = label_dict[technique_name]
            if score >= 0.0:
                label_vector[label_index] = 1.0

            # Example storing ranks for nDCG: rank 0 is the highest, rank 1 is next, etc.
            # For nDCG, you might want to store 'relevance' (1 if present).
            rank_dict[str(idx)] = label_index  # Or keep the same structure as before

        questions.append(question)
        labels_list.append(label_vector)
        ranks.append(rank_dict)

    return questions, labels_list, ranks

# -----------------------------------------------------------------------
# 4) Custom collate function
# -----------------------------------------------------------------------
def custom_collate_fn(batch):
    """
    For multi-label, labels come back as shape [num_classes].
    We can stack them into (batch_size, num_classes).
    """
    embeddings = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    ranks = [item[2] for item in batch]
    return embeddings, labels, ranks

# -----------------------------------------------------------------------
# 8) Multi-label MRR / nDCG (Optional Adaptation)
# -----------------------------------------------------------------------
def dcg(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0

def ndcg_at_k(predictions, labels, k=5):
    """
    predictions: shape (num_classes,) - predicted logits or probabilities
    labels: shape (num_classes,) - 0/1
    Both are for a single example.

    We sort the classes by descending predictions,
    then measure DCG of the ground truth over that ranking.
    """
    # Sort by descending prediction
    sorted_indices = np.argsort(-predictions)
    # Relevance is from the ground truth
    relevances = labels[sorted_indices]
    # compute nDCG
    ideal_relevances = np.sort(labels)[::-1]  # best-case
    dcg_val = dcg(relevances, k)
    idcg_val = dcg(ideal_relevances, k)
    return dcg_val / idcg_val if idcg_val != 0 else 0.0

def calculate_batch_ndcg(outputs, labels, k=5):
    """
    outputs: (batch_size, num_classes) raw logits
    labels: (batch_size, num_classes) 0/1
    """
    # Convert to CPU numpy
    outputs = torch.sigmoid(outputs).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    batch_ndcg = 0.0
    for i in range(len(outputs)):
        batch_ndcg += ndcg_at_k(outputs[i], labels[i], k)
    return batch_ndcg / len(outputs)

def calculate_batch_mrr(outputs, labels):
    """
    Multi-label MRR:
    We rank the classes by predicted probability, then for each relevant class,
    compute reciprocal rank. Then average across relevant classes for that sample.
    Finally, average over the batch.
    """
    outputs = torch.sigmoid(outputs).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    batch_mrr = 0.0
    for i in range(len(outputs)):
        preds = outputs[i]
        lbls = labels[i]
        # Descending order indices
        sorted_indices = np.argsort(-preds)

        # For each class that is relevant (lbls == 1), find rank
        relevant_indices = np.where(lbls == 1)[0]
        if len(relevant_indices) == 0:
            # If no relevant labels, skip or treat as 0
            continue

        rr_sum = 0.0
        for r in relevant_indices:
            rank = np.where(sorted_indices == r)[0][0] + 1  # 1-based rank
            rr_sum += 1.0 / rank

        # Average MRR for that sample
        sample_mrr = rr_sum / len(relevant_indices)
        batch_mrr += sample_mrr

    return batch_mrr / len(outputs)



# -----------------------------------------------------------------------
# 10) Final Evaluation on Test Set
# -----------------------------------------------------------------------
def evaluate_metrics_on_dataloader(model, dataloader):
    model.eval()
    total_samples = 0
    total_loss = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0

    with torch.no_grad():
        for embeddings_batch, labels_batch, _ in dataloader:
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(embeddings_batch)
            # print(outputs)

            loss = criterion(outputs, labels_batch)
            total_loss += loss.item()

            batch_size = embeddings_batch.size(0)
            total_mrr += calculate_batch_mrr(outputs, labels_batch) * batch_size
            total_ndcg += calculate_batch_ndcg(outputs, labels_batch, k=num_classes) * batch_size
            total_samples += batch_size

    avg_loss = total_loss / len(dataloader)
    avg_mrr = total_mrr / total_samples
    avg_ndcg = total_ndcg / total_samples
    print(f"Loss: {avg_loss:.4f}, MRR: {avg_mrr:.4f}, nDCG@{num_classes}: {avg_ndcg:.4f}")
    return avg_loss, avg_mrr, avg_ndcg

def evaluate_random_metrics(dataloader, num_classes, criterion):
    """
    Evaluate random predictions on MRR, nDCG, etc.
    """
    total_samples = 0
    total_loss = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0

    for embeddings_batch, labels_batch, _ in dataloader:
        batch_size = embeddings_batch.size(0)

        # Random 'logits' can just be random floats (unbounded)
        # shape: (batch_size, num_classes)
        random_logits = torch.randn(batch_size, num_classes)

        # We still compute loss using BCEWithLogitsLoss,
        # so move both predictions and labels to the same device
        # for consistency with your existing code, if needed:
        device = labels_batch.device
        random_logits = random_logits.to(device)
        labels_batch = labels_batch.to(device)

        loss = criterion(random_logits, labels_batch)
        total_loss += loss.item()

        # Calculate MRR and nDCG with your existing functions,
        # which expect a logits tensor:
        total_mrr += calculate_batch_mrr(random_logits, labels_batch) * batch_size
        total_ndcg += calculate_batch_ndcg(random_logits, labels_batch, k=num_classes) * batch_size

        total_samples += batch_size

    avg_loss = total_loss / len(dataloader)
    avg_mrr = total_mrr / total_samples
    avg_ndcg = total_ndcg / total_samples

    print(f"[Random Evaluation] Loss: {avg_loss:.4f}, MRR: {avg_mrr:.4f}, nDCG@{num_classes}: {avg_ndcg:.4f}")
    return avg_loss, avg_mrr, avg_ndcg


if __name__ == '__main__':
    set_seed(2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    file_path = '/PET-Select/PET_model_dataset/code_complex_classification_dataset_train.jsonl'
    # test_file_path = 'PET_model_dataset/code_complex_classification_dataset_test.jsonl'

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    # Replace your hardcoded line with this:

    test_file_path = f'/PET-Select/result/model_result_acc/{args.dataset}_{args.model}.jsonl'

    model_path = "/PET-Select/PET_model_result /code_complex_contrastive_model"

    data = list(map(json.loads, open(file_path)))
    test_data = list(map(json.loads, open(test_file_path)))

    # This dictionary maps technique names to indices
    label_dict = {
        'Zeroshot': 0,
        'Zeroshot_CoT': 1,
        'Fewshot': 2,
        'Fewshot_CoT': 3,
        'Persona': 4,
        'Self-planning': 5,
        'Self-refine': 6,
        'Progressive-Hint': 7,
        'Self-debug': 8
    }
    num_classes = len(label_dict)  # 9

    questions, multi_labels, ranks = generate_list(data)
    test_questions, test_multi_labels, test_ranks = generate_list(test_data)

    # Generate embeddings via SentenceTransformer
    embeddings = get_embedding(questions, model_path)
    test_embeddings = get_embedding(test_questions, model_path)

    # -----------------------------------------------------------------------
    # 5) Train/Val Split
    # -----------------------------------------------------------------------
    train_embeddings, eval_embeddings, train_labels, eval_labels, train_ranks, eval_ranks = train_test_split(
        embeddings, multi_labels, ranks, test_size=0.2, random_state=42
    )

    # Test set remains as is
    test_embeddings = test_embeddings
    test_labels = test_multi_labels
    test_ranks = test_ranks

    # -----------------------------------------------------------------------
    # 6) Create Datasets and Loaders
    # -----------------------------------------------------------------------
    batch_size = 16

    train_dataset = CustomDataset(train_embeddings, train_labels, train_ranks)
    eval_dataset = CustomDataset(eval_embeddings, eval_labels, eval_ranks)
    test_dataset = CustomDataset(test_embeddings, test_labels, test_ranks)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # -----------------------------------------------------------------------
    # 7) Initialize Model, Optimizer, Loss
    # -----------------------------------------------------------------------
    input_size = len(embeddings[0])  # e.g., 768 if using SBERT "base"
    model = MultiLabelClassificationModel(input_size, num_classes).to(device)

    # For multi-label, use BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 20

    # -----------------------------------------------------------------------
    # 9) Training Loop
    # -----------------------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for embeddings_batch, labels_batch, _ in train_dataloader:
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(embeddings_batch)  # (batch_size, num_classes)
            loss = criterion(outputs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_mrr = 0.0
        val_ndcg = 0.0
        with torch.no_grad():
            for val_embeddings_batch, val_labels_batch, _ in eval_dataloader:
                val_embeddings_batch = val_embeddings_batch.to(device)
                val_labels_batch = val_labels_batch.to(device)

                val_outputs = model(val_embeddings_batch)
                val_loss = criterion(val_outputs, val_labels_batch)
                val_running_loss += val_loss.item()

                # Evaluate MRR & nDCG on the batch
                val_mrr += calculate_batch_mrr(val_outputs, val_labels_batch) * val_embeddings_batch.size(0)
                val_ndcg += calculate_batch_ndcg(val_outputs, val_labels_batch,
                                                 k=num_classes) * val_embeddings_batch.size(0)

            val_loss_epoch = val_running_loss / len(eval_dataloader)
            # total samples in validation
            val_total = len(eval_dataset)
            val_mrr /= val_total
            val_ndcg /= val_total

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_loss_epoch:.4f} | "
              f"Val MRR: {val_mrr:.4f} | "
              f"Val nDCG: {val_ndcg:.4f}")

    print("\nEvaluating on Test Set...")
    evaluate_metrics_on_dataloader(model, test_dataloader)
    evaluate_random_metrics(test_dataloader, num_classes, criterion)

    # -----------------------------------------------------------------------
    # 11) Save Model
    # -----------------------------------------------------------------------
    output_model_path = '/PET-Select/PET_model_result /classification_model/multilabel_code_complex_classification_model_parameters2.pth'
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")


