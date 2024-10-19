import numpy as np
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from BCEmbedding import EmbeddingModel

def l2_normalize(embeddings):
    """对嵌入进行 L2 归一化"""
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def get_embeddings(sentences, model_name):
    """根据模型名称生成嵌入"""
    if model_name == 'm3e':
        model = SentenceTransformer('/data/czy/m3e-base')
        embeddings = model.encode(sentences)
    elif model_name == 'toolbench':
        model = SentenceTransformer('/data/czy/ToolBench_IR_bert_based_uncased')
        embeddings = model.encode(sentences)
    elif model_name == 'bge':
        model = BGEM3FlagModel('/data/czy/bge-m3', use_fp16=True)
        embeddings = model.encode(sentences, batch_size=12, max_length=8192)['dense_vecs']
    elif model_name == 'bce':
        model = EmbeddingModel(model_name_or_path="/data/czy/bce-embedding-base_v1")
        embeddings = model.encode(sentences)
    else:
        raise ValueError("Unsupported model name")

    return l2_normalize(embeddings)

def get_similarity(sentence1, sentence2, model_name):
    """返回两个句子之间的相似度，输入是两个句子和模型名称"""
    sentences = [sentence1, sentence2]
    embeddings = get_embeddings(sentences, model_name)
    similarity = float(embeddings[0] @ embeddings[1].T)  # 计算相似度
    return similarity

def main():
    # 示例使用
    sentence1 = 'Weather_forecast_v1'
    sentence2 = 'what is the weather like in Jiaxing right now?'
    
    model_names = ['m3e', 'bge', 'bce', 'toolbench']
    for model_name in model_names:
        similarity = get_similarity(sentence1, sentence2, model_name)
        print(f"Similarity between the sentences using {model_name.upper()}: {similarity:.4f}")

if __name__ == "__main__":
    main()