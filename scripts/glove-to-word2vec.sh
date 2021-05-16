for f in ../embeddings/glove.twitter.27B/*
  do python -m gensim.scripts.glove2word2vec --input "${f}" --output "${f}.word2vec"
done