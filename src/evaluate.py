import torch
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(test_dl, model, loss_fn, saved_path=None):
    if saved_path is not None:
        model.load_state_dict(torch.load(saved_path))

    model.eval()
    pds = []
    gts = []
    user_ids = []
    losses = []

    with torch.no_grad():
        for samples, labels, ids in test_dl:
            # fetch batch data
            labels = labels.to(device)
            samples = samples.to(device)
            preds = model(samples)

            losses.append(loss_fn(preds, labels).item())

            pds.append(preds.to("cpu").numpy())
            gts.append(labels.to("cpu").numpy())
            user_ids.append(ids.numpy())

    ndcg_df = pd.DataFrame({"user_id": np.squeeze(np.concatenate(user_ids)),
                            "preds": np.squeeze(np.concatenate(pds)),
                            "gts": np.squeeze(np.concatenate(gts))})

    def ndcg(data):
        if len(data) == 1:
            return 1

        return ndcg_score(np.asarray(data["gts"])[np.newaxis, :],
                          np.asarray(data["preds"])[np.newaxis, :])

    ndcg_scores = ndcg_df.groupby("user_id").apply(ndcg)
    mean_ndcg = np.mean(ndcg_scores)
    mean_loss = np.mean(losses)
    return mean_ndcg, mean_loss

