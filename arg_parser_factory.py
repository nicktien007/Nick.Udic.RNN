from argparse import ArgumentParser, RawTextHelpFormatter


def build():
    parser = ArgumentParser(
        description='訓練 RNN 神經網路'
        , formatter_class=RawTextHelpFormatter)

    subcmd = parser.add_subparsers(
        dest='subcmd', help='subcommands', metavar='SUBCOMMAND')
    subcmd.required = True

    # 進行RNN 訓練
    train_parser = subcmd.add_parser('train',
                                     help='訓練 RNN 神經網路')
    train_parser.add_argument('-i',
                              dest='input',
                              help='解析的檔案路徑')
    train_parser.add_argument('-cs','--corpusSize',
                              dest='corpus_size',
                              default=60000,
                              help='取多少筆資料訓練')
    train_parser.add_argument('-t', '--temp',
                              dest='temp',
                              default='./temp',
                              help='輸出 trainData 檔案路徑')
    train_parser.add_argument('-o',
                              dest='output',
                              help='輸出 trainModel 檔案路徑')
    train_parser.add_argument('-d', '--device',
                              dest='device',
                              default='cpu',
                              help='cpu or cuda')

    # 進行查詢
    query_parser = subcmd.add_parser('query',
                                     help='進行查詢')
    query_parser.add_argument('-i',
                              dest='input',
                              help='進行【查詢】的檔案路徑')
    query_parser.add_argument("-k", '--keyword',
                              dest='keyword',
                              help='關鍵字',
                              default="")
    query_parser.add_argument('-d', '--device',
                              dest='device',
                              default='cpu',
                              help='cpu or cuda')

    return parser.parse_args()
