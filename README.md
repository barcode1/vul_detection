The goal of this project is to detect XSS SQLI, Command Injection, and Zero Day vulnerabilities with advanced linguistic and deep models.
This is a pipeline model with the following model architecture:


1- First, semantic embeddings are combined with secbert, word2vec, and fasttext to enrich the text and semantic relevance.

2- After combining the embeddings, the malicious keywords are weighted with the attention algorithm.

3- Then the weighted output is given to the input of the CNN-BILSTM model for feature extraction.

4- The output of the previous stage is applied to two anomaly detection branches for detecting unknown zero-day vulnerabilities and the Codebert classifire branch as input to the branches.

5- In the last stage, an output is generated using the model ensemble and calculating the probability of each branch.

6- Testing in other LLMs in future.
