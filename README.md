# EGNN_loma

An implementation of [EGNN](https://arxiv.org/abs/2102.09844) in [Loma](https://github.com/BachiLi/loma_public)

### Get Start

Simply run the following script to train a Loma EGNN:

```shell
pip install -r requirements.txt
cd egnn
python run_egnn.py
```

The default complied backend is OpenCL. If you want to use other backends, please change the **TARGET** variable in **egnn/ops/loma/\_\_init\_\_.py**
