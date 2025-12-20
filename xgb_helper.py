

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnn_helper import *
import xgboost as xgb

import umap
from sklearn.manifold import TSNE

def predict_sequence_xg_boost(one_hot, embedding_model, xgb_model):
    xgb.set_config(verbosity=0)
    embed = embedding_model.predict(one_hot)

    pred_norm = xgb_model.predict(embed)[0]
    
    return pred_norm

def plot_predictions_xg_boost(total_x_vals, COMBINED_DF, path, embedding_model, xgb_model):
    temp_y_vals = []
    temp_pred_vals = []

    temp_x_vals = []
    for i in range(total_x_vals):
        temp_y_vals.append(COMBINED_DF.iloc[i, 1])
        temp_x_vals.append(i)
        seq = COMBINED_DF.iloc[i,  0]
        temp_batch_seq = convert_to_one_hot(str(seq))[np.newaxis, ...]
    
        pred = predict_sequence_xg_boost(temp_batch_seq, embedding_model, xgb_model)
        temp_pred_vals.append(pred)
    
    rmse = np.sqrt(np.mean((np.array(temp_y_vals) - np.array(temp_pred_vals))**2))
    mae = np.mean(np.abs(np.array(temp_y_vals) - np.array(temp_pred_vals)))


    plt.figure(figsize=(15, 5))
    plt.plot(temp_x_vals, temp_y_vals, label='actual', linestyle='--', color='blue')
    plt.plot(temp_x_vals, temp_pred_vals, label='preds', linestyle='-', color='red')
    plt.ylabel(f'normalized indel freq')
    plt.xlabel(f'RMSE {rmse: .4f} | MAE {mae: .4f}')
    plt.savefig(path)


def graph_xgb_model_history(history, path):
    #loss vals
    val = history['validation_0']['rmse']

    plt.figure(figsize=(12, 4))
    plt.plot(val)
    plt.title('model rmse')
    plt.ylabel('rmse')
    plt.xlabel('estimators')
    plt.legend(['val'])

    plt.savefig(path)


def tnse_embedding_visualization(x_train_embed, y_train):
    tsne = TSNE(
    n_components=2,
    learning_rate='auto',
    init='random',
    perplexity=30
    )
    x_train_tsne = tsne.fit_transform(x_train_embed)


    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        x_train_tsne[:, 0], x_train_tsne[:, 1],
        c=y_train, cmap='viridis', s=12, alpha=0.8
    )

    plt.colorbar(scatter, label="Normalized Indel Frequency")
    plt.title("t-SNE of CNN Embeddings (GlobalMaxPool Layer)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig("cnn_xgb_graphs/tsne_embedding_visualization.png")


def umap_embedding_visualization(embeddings, y):
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )

    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=y,
        cmap='viridis',
        s=5
    )
    plt.colorbar(label='Normalized Indel Frequency')
    plt.title("UMAP of CNN Embeddings")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig("cnn_xgb_graphs/umap_embedding_visualization.png")
