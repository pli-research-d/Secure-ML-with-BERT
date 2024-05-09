# Analysis, Modeling and Design of Personalized Digital Learning Environment: A Proof of Concept

This research analyzes, models and develops a novel Digital Learning Environment (DLE) fortified by the innovative Private Learning Intelligence (PLI) framework. The proposed PLI framework leverages federated machine learning (FL) techniques to autonomously construct and continuously refine personalized learning models for individual learners, ensuring robust privacy protection. Our approach is pivotal in advancing DLE capabilities, empowering learners to actively participate in personalized real-time learning experiences. The integration of PLI within a DLE also streamlines instructional design and development demands for personalized teaching/learning. We seek ways to establish a foundation for the seamless integration of FL into teaching/learning systems, offering a transformative approach to personalized learning in digital environments.

## Proposed Architecture
![fig3](PLI Architecture.png}
## Implementation Details

1. **Required Imports and Environment Setup**
    - The following libraries are required for the project:
      - `torch`, `transformers`, `torch.nn.functional`, `torch.utils.data`
      - `numpy`, `cryptography.fernet`

    ```python
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch.nn.functional as F
    import numpy as np
    from cryptography.fernet import Fernet
    ```

2. **Secure ML Environment Class**
    - Defines a class to sandbox data securely:
    ```python
    class SecureMLEnvironment:
        def __init__(self):
            self.sandbox = {}

        def isolate(self, data):
            data_id = f"data_{len(self.sandbox)}"
            self.sandbox[data_id] = data
            return data

    secure_env = SecureMLEnvironment()
    ```

3. **Privacy Safeguard Class**
    - A class for anonymizing and encrypting data:
    ```python
    class PrivacySafeguard:
        key = Fernet.generate_key()
        cipher = Fernet(key)

        @staticmethod
        def anonymize(data):
            return np.array(['ANONYMIZED' for _ in data])

        @staticmethod
        def encrypt_data(data):
            return [PrivacySafeguard.cipher.encrypt(str(item).encode()).decode() for item in data]
    ```

4. **Learning Dataset Class**
    - A custom dataset class to manage tokenization and encryption:
    ```python
    class LearningDataset(Dataset):
        def __init__(self, data, targets, tokenizer, max_length=512):
            data = PrivacySafeguard.anonymize(PrivacySafeguard.encrypt_data(data))
            self.data = data
            self.targets = targets
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            label = self.targets[idx]
            tokens = self.tokenizer(
                f"{item[0]} {item[1]} {item[2]} {item[3]} {item[4]} {item[5]} {item[6]} {item[7]} {item[8]}",
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            return {
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float32)
            }
    ```

5. **Model Setup**
    - Loads a BERT tokenizer and model for classification:
    ```python
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    ```

6. **Dataset and DataLoader Setup**
    - Prepares training data and defines data loaders:
    ```python
    training_data = secure_env.isolate(np.array([
        [3, 8, 10, 4, 78.0, 85.0, 3, 1, 6],
        [5, 12, 15, 8, 82.0, 89.0, 4, 0, 7]
    ]))

    target_scores = np.array([
        [6.0, 7.5, 8.0, 7.0],
        [8.0, 9.0, 9.5, 9.0]
    ])

    dataset = LearningDataset(training_data, target_scores, tokenizer)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    ```

7. **Training Loop**
    - Optimizes model parameters and evaluates on a validation dataset:
    ```python
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        best_val_loss = float('inf')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained('best-learning-model')
            tokenizer.save_pretrained('best-learning-model')
    ```

8. **Inference Function**
    - Evaluates new data based on the trained model:
    ```python
    def calculate_learning_measures(logins, time_spent, page_visits, search_queries, activity_completion, quiz_score, reactions_pos, reactions_neg, feedback):
        input_text = f"{logins} {time_spent} {page_visits} {search_queries} {activity_completion} {quiz_score} {reactions_pos} {reactions_neg} {feedback}"
        tokens = tokenizer(input_text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            output = model(**tokens)
            scores = torch.sigmoid(output.logits).squeeze(0).cpu().numpy()

        return {
            'Conscientiousness': round(scores[0] * 10, 2),
            'Motivation': round(scores[1] * 10, 2),
            'Understanding': round(scores[2] * 10, 2),
            'Engagement': round(scores[3] * 10, 2)
        }
    ```

9. **Example Usage**
    - Tests the inference function with example data:
    ```python
    learning_scores = calculate_learning_measures(4, 9, 11, 5, 80.0, 84.0, 3, 2, 6)
    print("Learning Measures Scores:", learning_scores)
    ```

## Getting Started

1. **Install Dependencies**:
   - Make sure you have `torch`, `transformers`, and `cryptography` installed.

2. **Run the Training Script**:
   - Train the BERT model with `SecureMLEnvironment`.

3. **Use the Inference Function**:
   - Classify new inputs using `calculate_learning_measures`.

## Further Information

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transformers Library](https://huggingface.co/docs/transformers/index)
- [Cryptography Documentation](https://cryptography.io/en/latest/)

