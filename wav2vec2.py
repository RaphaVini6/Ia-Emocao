from datasets import load_dataset, Audio
import os
from transformers import Wav2Vec2ForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
import numpy as np
import evaluate
os.environ["USE_TF"] = "0" 

class AudioDataCollator:
    def __init__(self, processor, padding="longest"):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_values = [feature["input_values"] for feature in features]
        labels = [feature["label"] for feature in features]

        batch = self.processor.pad(
            {"input_values": input_values},
            padding=self.padding,
            return_tensors="pt"
        )

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

ds = load_dataset("Hemg/Emotion-audio-Dataset")

ds = ds.cast_column("audio", Audio())
ds = ds["train"].train_test_split(test_size=0.1)

print(ds)

from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=7)

def preprocess_data(batch):
    audio = batch["audio"]["array"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    batch["input_values"] = inputs.input_values[0]
    return batch

ds = ds.map(preprocess_data, remove_columns=["audio"])

metric = evaluate.load("accuracy")

def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=pred.label_ids)


data_collator = AudioDataCollator(processor)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16, #modificar para 8 caso memória se torne um problema
    per_device_eval_batch_size=16, #modificar para 8 caso memória se torne um problema
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir=None
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_results = trainer.evaluate()
print(f"Resultados de avaliação: {eval_results}")

model.save_pretrained("./emotion-recognition-model")
processor.save_pretrained("./emotion-recognition-model")
