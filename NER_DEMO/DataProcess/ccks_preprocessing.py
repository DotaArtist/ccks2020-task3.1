"""
人民日报数据预处理

数据已经被初步处理了。现在数据格式为:
    中/B_nt 共/M_nt 中/M_nt 央/E_nt 总/O 书/O 记/O 、/O
    国/O 家/O 主/O 席/O 江/B_nr 泽/M_nr 民/E_nr

里面的实体有：人名、地名、机构名称
    人名： B_nr M_nr E_nr
    地名： B_ns M_ns E_ns
    机构名称: B_nt M_nt E_nt

由于这个数据的处理需要处理成:
人名：
    B_nr -> B-PER
    M_nr -> I-PER
    E_nr -> I-PER

地名：
    B_ns -> B-LOC
    M_ns -> I-LOC
    E_ns -> I-LOC

机构名称:
    B_nt -> B-ORG
    M_nt -> I-ORG
    E_nt -> I-ORG


将renmin3.txt 文件分割成两份分别为训练集和测试集，分割比例按照8:2进行分割。
分割后存储名称为: train.txt   test.txt

存储格式:(每个字符和标志为一行，字符和标志空格符号隔开。每一句用回车隔开)
char tag
树 O
立 O
企 O
业 O
形 O
象 O
的 O
需 O
要 O

大 B-ORG
连 I-ORG
海 I-ORG
富 I-ORG
集 I-ORG
团 I-ORG

"""

from Public.path import path_mydata_dir
import os


# 读取数据
def read_file(file_path: str) -> [str]:
    with open(file_path, 'r') as f:
        texts = f.read().split('\n')
    return texts

def ccks_preprocessing(split_rate: float = 0.8,
                        ignore_exist: bool = False) -> None:
    """
    data2数据预处理
    :param split_rate: 训练集和测试集切分比例
    :param ignore_exist: 是否忽略已经存在的文件(如果忽略，处理完一遍后不会再进行第二遍处理)
    :return: None
    """
    path = os.path.join(path_mydata_dir, "all.txt")
    path_train = os.path.join(path_mydata_dir, "train.txt")
    path_test = os.path.join(path_mydata_dir, "test.txt")

    if not ignore_exist and os.path.exists(path_train) and os.path.exists(path_test):
        return

    texts = []
    with open(path, 'r') as f:
        line_t = []
        for l in f:
            if l != '\n':
                line_t.append(l)
            else:
                texts.append(line_t)
                line_t = []

    if split_rate >= 1.0:
        split_rate = 0.8
    split_index = int(len(texts) * split_rate)
    train_texts = texts[:split_index]
    test_texts = texts[split_index:]

    # 分割和存数文本
    def split_save(texts: [str], save_path: str) -> None:
        data = []
        for line in texts:
            for item in line:
                data.append(item)
            data.append("\n")
        with open(save_path, 'w') as f:
            f.write("".join(data))

    split_save(texts=train_texts, save_path=path_train)
    split_save(texts=test_texts, save_path=path_test)


if __name__ == '__main__':
    ccks_preprocessing()
