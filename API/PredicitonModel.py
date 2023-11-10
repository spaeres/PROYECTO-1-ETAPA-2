import joblib

class Model:

    def __init__(self):
        self.model = joblib.load("assets/tfidf_model.joblib")
        self.tfidf = joblib.load("assets/tfidf_transform.pkl")

    def make_predictions(self, data):
        # Se transforman los datos con el transformador de TF-IDF:
        self.tfidf.transform(data)
        result = self.model.predict(data)
        return result
