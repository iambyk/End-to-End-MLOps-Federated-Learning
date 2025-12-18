import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from utils import load_and_partition_data

data = load_and_partition_data("data/creditcard.csv", num_clients=3)
(X_train, y_train, X_test, y_test) = data[0] 

model = LogisticRegression(max_iter=1)

try:
    idx_0 = np.where(y_train == 0)[0][0]
    idx_1 = np.where(y_train == 1)[0][0]
    model.fit(X_train[[idx_0, idx_1]], y_train[[idx_0, idx_1]])
    print("Model başarıyla iki sınıfla ilklendirildi.")
except IndexError:
    print("HATA: Bu bankanın verisinde dolandırıcılık örneği (1) bulunamadı!")
    print("Lütfen utils.py dosyasındaki Stratified split kısmını kontrol et.")

class FraudClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        if hasattr(model, "coef_"):
            return [model.coef_, model.intercept_]
        else:
            return []

    def fit(self, parameters, config):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        model.fit(X_train, y_train)
        print("Banka üzerinde eğitim tamamlandı...")
        return [model.coef_, model.intercept_], len(X_train), {}

    def evaluate(self, parameters, config):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        print(f"Test Başarımı: {accuracy}")
        return loss, len(X_test), {"accuracy": accuracy}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FraudClient())