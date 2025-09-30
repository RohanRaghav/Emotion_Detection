from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model.save_pretrained("./bert_local")
tokenizer.save_pretrained("./bert_local")
