# figocr
CRNN神经网络用于可变长度的手写数字识别，ocr2.pth是预训练模型。
下载链接: https://pan.baidu.com/s/17wSEXvObrhMvoVnXCmewdg 提取码: mt7u

另外：

src/procedure 目录包含了实用的图片处理函数，比如查找定位点，图片对齐等等。

#### 执行
```bash
python src/main.py --model=ocr2.pth --image=examples/31280.png
```
#### 输出
```
INFO:root:model loaded from ocr2.pth
INFO:root:3------1-2-8-0------
31280
```
![](https://github.com/HuangFJ/figocr/blob/master/examples/31280.png)
