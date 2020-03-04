# figocr
crnn模型可变长度的手写数字识别，ocr2.pth是预训练模型。
下载链接: https://pan.baidu.com/s/17wSEXvObrhMvoVnXCmewdg 提取码: mt7u

#### 执行
```bash
python src/main.py --model=ocr2.pth --image=examples/31280.png
```
#### 输出
```
INFO:root:model loaded from src/model/ocr2.pth
INFO:root:3------1-2-8-0------
31280
```
![](https://github.com/HuangFJ/figocr/blob/master/examples/31280.png)