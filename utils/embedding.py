import numpy as np

def make_embedding(in_path, save_path):
    transe = np.load(in_path)
    embeding_size = transe.shape[1]
    pad = np.zeros([1, embeding_size])
    embedding = np.concatenate([transe, pad])
    print(embedding.shape)
    np.save(save_path, embedding)

if __name__ == "__main__":
    make_embedding("../data/semmed/transe.embedding.ent.npy", "../data/semmed/cui_embedding.npy")