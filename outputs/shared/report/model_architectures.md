## Model Architectures

### Unified Comparison

| Model        | Input Type         | Architecture                      | Params    | Trainable | Embed Dim | Dropout | Hidden Dim |
| ------------ | ------------------ | --------------------------------- | --------- | --------- | --------- | ------- | ---------- |
| TF-IDF + MLP | TF-IDF vectors     | 2-layer MLP (1 hidden layer)      | 5,197,975 | 5,197,975 | N/A       | 0.2     | 512        |
| Text CNN     | Token-ID sequences | Multi-kernel CNN (3 kernel sizes) | 1,930,167 | 1,930,167 | 256       | 0.5     | N/A        |
| BiLSTM       | Token-ID sequences | 2-layer bidirectional LSTM        | 4,284,311 | 4,284,311 | 256       | 0.3     | 256        |

### TF-IDF + MLP

Input: 10,000 TF-IDF features (10,000-dimensional). Single hidden layer with 512 units, ReLU activation, dropout 0.2.

### Text CNN

Embedding dimension: 256. Kernel sizes: [3, 4, 5] with 100 filters each. ReLU activation, max-over-time pooling, dropout 0.5. Trainable embeddings.

### BiLSTM

Embedding dimension: 256. Hidden dimension: 256, 2 layers, bidirectional. Summarization: concat_final_hidden. Gradient clipping (max norm 1.0). Dropout 0.3. Trainable embeddings.
