# Assuming you already have Transformer outputs and attention weights
import matplotlib.pyplot as plt
import numpy as np

def VizAttention(attention_weights, input_sentence, output_sentence):
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(attention_weights, cmap='bone')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + input_sentence.split(), rotation=90)
    ax.set_yticklabels([''] + output_sentence.split())

    plt.show()
