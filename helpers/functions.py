import pandas as pd
from helpers import model


def get_model_response(json_data):
    X = pd.DataFrame.from_dict(json_data)
    prediction = model.predict(X.text)
    if int(prediction) == 1:
        label = "Yes"
    else:
        label = "No"
    return {
        'status': 200,
        'label': label,
        'prediction': int(prediction)
    }
