"""
Use this script to collect samples from the models (baseline models).
    for set in ["train", "validation", "test"]:
    - Create a support set (5 class, 5 examples)
    - Create a query set (5 examples)
    - Feed each query feed it to the model (5 times)
    - obtain logits for each sample (25 logits)
    - Get top10.
In your plots show the class labels and logits.
"""