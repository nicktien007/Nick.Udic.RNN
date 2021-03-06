# Nick.Udic.RNN
> Pytorch 搭建RNN神經網路分類句子



依賴套件：
- pytorch：機器學習庫([install](https://pytorch.org/))
- opencc：簡、繁轉換
- jieba：分詞
- numpy
- Pandas

```
pip install jieba
pip install opencc
pip install numpy
pip install pandas
```

## 訓練 RNN 神經網路

參數
- **-i**：待訓練的資料路徑
- **-cs**：取多少筆資料訓練(`預設：60000`)
- **-t**：輸出 trainData 檔案路徑(`預設：./temp`)
- **-o**：輸出的Model路徑
- **-d**：cpu or cuda(`預設：cpu`)
```
python3 main.py train -i ./dataset/online_shopping_10_cats.csv -o ./test_model
```

## query 進行關鍵字查詢
參數
- **-i**：待查詢的Model路徑
- **-k**：查詢句子分類
- **-d**：cpu or cuda(`預設：cpu`)
```
python3 main.py query -i ./trained_model -k 這東西很糟，我很不爽
```

## 參考資料
- https://bit.ly/35PPUK3
- https://bit.ly/2HRXbRe
- https://bit.ly/34PAvu5
- https://bit.ly/34M7r6p
- https://bit.ly/3kOipy3
- https://bit.ly/3oYIvAU