import numpy as np
class CustomDeque:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.data = []

    def append(self, value):
        """在队列尾部添加元素"""
        if len(self.data) >= self.maxlen:
            self.data.pop(0)  # 删除最早添加的元素！
        self.data.append(value)

    def append_left(self, value):
        """在队列头部添加元素"""
        if len(self.data) >= self.maxlen:
            self.data.pop()  # 删除最左端的元素！
        self.data.insert(0, value)

    def pop(self):
        """移除并返回队列尾部的元素"""
        if self.data:
            return self.data.pop()
        raise IndexError("笨蛋！空队列删除个屁的元素！")

    def pop_left(self):
        """移除并返回队列头部的元素"""
        if self.data:
            return self.data.pop(0)
        raise IndexError("笨蛋！空队列删除个屁的元素！")

    def __getitem__(self, index):
        # 支持通过索引访问元素
        return self.data[index]

    def __repr__(self):
         return repr(self.data)