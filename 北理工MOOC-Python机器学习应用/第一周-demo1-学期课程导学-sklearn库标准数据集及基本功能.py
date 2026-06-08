"""
@File: 第一周-demo1-学期课程导学-sklearn库标准数据集及基本功能.py
@Author: cuishuohao
@Date: 2026/5/29 01:14
"""

# 鸢尾花数据集——加载示例
def func1():
    from sklearn.datasets import load_iris
    iris = load_iris()
    print(iris.data.shape)
    print(iris.target.shape)
    print(list(iris.target_names))
    print(iris.target_names)

    data,target = load_iris(return_X_y=True)
    print(data.shape)
    print(target.shape)


# 手写数字数据集
def func2():
    from sklearn.datasets import load_digits
    digits = load_digits()
    print(digits.data.shape)
    print(digits.target.shape)
    print(digits.images.shape)
    import matplotlib.pyplot as plt
    plt.matshow(digits.images[0])
    plt.show()
    print(digits.target_names)


func1()
# func2()