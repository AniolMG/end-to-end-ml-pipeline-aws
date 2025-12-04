import pandas as pd
from model_loader import load_model_and_encoders


class Predictor:
    def __init__(self):
        self.model, self.encoders = load_model_and_encoders()

    def _prepare_df(self, instances):
        df = pd.DataFrame(instances)
        df = df.fillna(0)
        return df

    def _apply_encoders(self, df):
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = encoder.transform(df[col])
        return df

    def predict(self, instances):
        df = self._prepare_df(instances)
        df = self._apply_encoders(df)
        preds = self.model.predict(df)
        return preds.tolist()
