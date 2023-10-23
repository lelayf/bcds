import replicate
from sklearn.metrics import classification_report
import pandas as pd

df_train = pd.read_csv('../data/Ask0729-fixed.txt', sep='\t', header=None, names=["label", "text"])
df_test = pd.read_csv('../data/testSet-qualifiedBatch-fixed.txt', sep='\t', header=None, names=["label", "text"])
df_train['target'] = df_train.apply(lambda r: 1 if r['label']=='Yes' else 0, axis=1)
df_test['target'] = df_test.apply(lambda r: 1 if r['label']=='Yes' else 0, axis=1)


embeddings = []
for text in list(df_train.text):

    output = replicate.run(
        "replicate/all-mpnet-base-v2:b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305", 
        input={"text": text}, 
    )
    embeddings.append(output[0]['embedding'])
    print(output[0]['embedding'])

colnames = ["c" + str(i) for i in range(768)]
emb_df = pd.DataFrame(embeddings, columns=colnames)
emb_df.to_parquet('../data/train_embeddings.pq')

#     pred = 0         
#     if answer == 'followup':
#         pred = 1

#     preds.append(pred)

#     print(p, "CLASS:", pred)

# preds_df = pd.DataFrame({"preds":preds})
# preds_df.to_parquet("../mistral_predictions.pq")

# print(classification_report(df_test.target,preds))



