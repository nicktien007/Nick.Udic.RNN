import torch
import logging_utils

from service import train_service
from service.tokenize_service import tokenize
from service.train_service import build_RNN_model, setup_manual_seed
from service.predict_service import predict


def main():
    # preprocess_dataset("./dataset/online_shopping_10_cats.csv", "./train_data")
    # build_RNN_model('./train_data', './trained_model',torch.device('cpu'))
    predict("./trained_model", "我非常快樂")


# https://bit.ly/35PPUK3
# https://bit.ly/2HRXbRe
# https://bit.ly/34PAvu5
# https://bit.ly/34M7r6p
# https://bit.ly/3kOipy3
if __name__ == '__main__':
    logging_utils.Init_logging()
    main()
