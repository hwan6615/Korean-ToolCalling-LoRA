from datasets import load_dataset

ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
print(ds)
print(ds[0])