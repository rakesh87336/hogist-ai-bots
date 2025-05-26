from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Specify the model name
model_name = "google/flan-t5-small"

# Load tokenizer and model from Hugging Face hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Save locally to a directory named 'local_flan_t5_small'
save_directory = "local_flan_t5_small"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model and tokenizer saved to: {save_directory}")
