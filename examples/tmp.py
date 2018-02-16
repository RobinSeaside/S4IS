#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 201802 LIU Wangsheng; Email: awang.signup@gmail.com
# 最大连续子序列
def findSub(numbers):
    sub_sum = 0
    max_sub_sum = 0
    start = 0
    end = 0
    for i, number in enumerate(numbers):
        sub_sum = sub_sum + number
        if sub_sum > max_sub_sum:
            max_sub_sum = sub_sum
            end = i
        if sub_sum <= 0:
            sub_sum = 0
            start = i + 1
    return max_sub_sum, numbers[start:end+1]

print(findSub([-1,2,5,3,-1,-2,4,1,-5]))

# 数列中大于一半的元素
def findBoss(numbers):
    boss = 0
    counter = 0
    for i in range(len(numbers)):
        if counter == 0:
            boss = numbers[i]
            counter = 1
        elif numbers[i] == boss:
            counter += 1
        else:
            counter -= 1
    if counter > 0:
        return boss
    else:
        return None

print(findBoss([3, 2, 3, 1, 3, 3, 2, 3]))
print(findBoss([3, 2, 2, 1, 3, 3, 2, 2]))


# 无序整数数组，3个数乘积最大

# 旋转数组找出最小数字

# 第k大的数

# 前序遍历、反转链表

# 找出一个整数可能是哪些质数对的和