# Analysis, Modeling and Design of Personalized Digital Learning Environment: A Proof of Concept

This research analyzes, models and develops a novel Digital Learning Environment (DLE) fortified by the innovative Private Learning Intelligence (PLI) framework. The proposed PLI framework leverages federated machine learning (FL) techniques to autonomously construct and continuously refine personalized learning models for individual learners, ensuring robust privacy protection. Our approach is pivotal in advancing DLE capabilities, empowering learners to actively participate in personalized real-time learning experiences. The integration of PLI within a DLE also streamlines instructional design and development demands for personalized teaching/learning. We seek ways to establish a foundation for the seamless integration of FL into teaching/learning systems, offering a transformative approach to personalized learning in digital environments.

## Proposed Architecture
![fig3](https://github.com/pli-research-d/Secure-ML-with-BERT/blob/069015f8bbc4271a644b3b94cbf3bf0e6ec7f3b3/PLI%20Architecture.png)

We enter the realm of PLI, where innovation meets the necessity to address two critical challenges head-on. Firstly, we tackle the imperative for dynamic, locally-trained personal ML model training, safeguarding sensitive data within the user's environment. No longer will privacy concerns hinder progress; PLI ensures our data stays where it belongs—under our control.

PLI goes beyond mere privacy protection; it pioneers the seamless integration of local personal models and global knowledge models. Imagine unlocking a world of personalized, domain-specific insights directly within our learning experience. With PLI, education becomes more than just acquiring knowledge—it's about harnessing insights tailored to our unique journey and aspirations. We aim to revolutionize learning through the power of PLI, where privacy and personalization converge to shape a brighter future.

Following implementation demonstrates a secure and personalized learning environment using various techniques and libraries.

## Key Aspects

### Secure Environment
- **SecureMLEnvironment**: Isolates data within sandboxes for secure handling using unique identifiers.
- **PrivacySafeguard**: Provides data anonymization and encryption using Fernet encryption.

### Learning Dataset
- **LearningDataset**: Custom dataset class, handling data loading, anonymization, encryption, and tokenization for machine learning tasks.

### Model Setup
- **BertTokenizer**: Tokenizes text data using the transformers library.
- **BertForSequenceClassification**: Builds a sequence classification model based on the BERT architecture from the transformers library.

### Data Loading and Preparation
- Loads data from a CSV file using pandas.
- Prepares data by anonymizing, encrypting, and tokenizing it.
- Splits data into training and validation sets.

### Training and Validation
- **train_and_validate**: Function to train the model on the training data and validate it on the validation data.
  - Uses the AdamW optimizer.
  - Tracks training and validation loss.
  - Saves the model with the best validation loss.

### Hyperparameter Tuning
- **hyperparameter_tuning**: Function using HyperOpt to find the best hyperparameters.
  - Defines a search space for learning rate, batch size, and epochs.
  - Trains the model with different hyperparameter combinations and selects the one with the best validation loss.

### Inference and Learning Measures
- **calculate_learning_measures**: Function that calculates learning scores from user interaction data using the trained model.
  - Uses the sigmoid function to convert model output into scores between 0 and 1.

### Model Averaging
- Functions to load and average multiple models.
- Demonstrates how to average a personalized model with a global model.

### Contextual Chatbot
- **contextual chatbot example**: Class that uses a large language model (LLM) adapter from Gradient AI to generate responses.
  - Allows personalization based on the user's learning scores.
  - Extracts topics from user prompts using spaCy and Rake.
  - Implements a chat loop that handles user input, generates responses, and displays learning scores and topic information.


## Implementation Details

1. **Required Imports and Environment Setup**
    - The following libraries are required for the project:
      - `torch`, `transformers`, `torch.nn.functional`, `torch.utils.data`
      - `scikit-learn`, `panda`, `numpy`, `cryptography.fernet`, `hyperopt`

    ```python
    import torch
    import pandas as pd
    import numpy as np
    from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
    from sklearn.metrics import accuracy_score
    from transformers import BertTokenizer, BertForSequenceClassification
    from cryptography.fernet import Fernet
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
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
                ' '.join(map(str, item)),
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
    csv_file_path = './sample_data/learner_behavior_data.csv'
    
    def load_data(file_path):
        df = pd.read_csv(file_path, chunksize=1000)
        data = []
        targets = []
        for chunk in df:
            data.extend(chunk.iloc[:, :-4].values)
            targets.extend(chunk.iloc[:, -4:].values)
        data_id, isolated_data = secure_env.isolate(data)
        return isolated_data, targets
    
    training_data, target_scores = load_data(csv_file_path)
    
    dataset = LearningDataset(training_data, target_scores, tokenizer)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    ```

7. **Training and Validation Loop**
    - Optimizes model parameters and evaluates on a validation dataset:
    ```python
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def train_and_validate(model, train_loader, val_loader, device, num_epochs=1):
        personalized_val_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
    
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
    
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
    
            avg_train_loss = total_train_loss / len(train_loader)
    
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()
    
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
            if avg_val_loss < personalized_val_loss:
                personalized_val_loss = avg_val_loss
                model.save_pretrained('personalized-learning-model')
                tokenizer.save_pretrained('personalized-learning-model')
    ```

8. **Hyperparameter Tuning Function**
    - Performs hyperparameter tuning for the machine learning model:
    ```python
    def hyperparameter_tuning(data, targets, tokenizer, max_length=512):
    
        def objective(params):
            dataset = LearningDataset(data, targets, tokenizer, max_length)
            train_size = int(0.75 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
    
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    
            best_val_loss = float('inf')
            for epoch in range(params['epochs']):
                model.train()
                total_train_loss = 0
    
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
    
                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
    
                avg_train_loss = total_train_loss / len(train_loader)
    
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
    
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        total_val_loss += loss.item()
    
                avg_val_loss = total_val_loss / len(val_loader)
    
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
    
            return {'loss': best_val_loss, 'status': STATUS_OK}
    
        search_space = {
            'learning_rate': hp.loguniform('learning_rate', -5, -3),
            'batch_size': hp.choice('batch_size', [2, 4, 8]),
            'epochs': hp.choice('epochs', [3, 5, 10])
        }
    
        trials = Trials()
        best_params = fmin(objective, search_space, algo=tpe.suggest, max_evals=1, trials=trials)
  
        print('Best hyperparameters:', best_params)
        print('Best validation loss:', trials.best_trial['result']['loss'])
    
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'])
        train_and_validate(model, train_loader, val_loader, device, num_epochs=best_params['epochs'])
    
        return model, best_params  # Return both model and hyperparameters
    
    model, best_params = hyperparameter_tuning(training_data, target_scores, tokenizer)
    ```
    
9. **Inference Function**
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

10. **Example Usage**
    - Tests the inference function with example data:
    ```python
    learning_scores = calculate_learning_measures(4, 9, 11, 5, 80.0, 84.0, 3, 2, 6)
    print("Learning Measures Scores:", learning_scores)
    ```

## Video 
**Screen recording of the process**

[![YouTube](http://i.ytimg.com/vi/LH0zSi6c7JQ/hqdefault.jpg)](https://www.youtube.com/watch?v=LH0zSi6c7JQ)

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

