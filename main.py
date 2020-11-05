import torch

import arg_parser_factory
import logging_utils

from service.preprocess_dataset_service import preprocess_dataset
from service.train_service import build_RNN_model
from service.predict_service import predict


def main():
    args = arg_parser_factory.build()

    if args.subcmd == 'train':
        preprocess_dataset(args.input, args.temp, int(args.corpus_size))
        build_RNN_model(args.temp, args.output, torch.device(args.device))

    if args.subcmd == 'query':
        predict(args.input, args.keyword, torch.device(args.device))


# https://bit.ly/35PPUK3
# https://bit.ly/2HRXbRe
# https://bit.ly/34PAvu5
# https://bit.ly/34M7r6p
# https://bit.ly/3kOipy3
# https://bit.ly/3oYIvAU
if __name__ == '__main__':
    logging_utils.Init_logging()
    main()
