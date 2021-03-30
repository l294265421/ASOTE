import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


def plot_attention(words, attention):
    """

    :param words:
    :param attention:
    :return:
    """
    data = {}
    for i in range(len(words)):
        data[words[i]] = {'attention': attention[i]}
    d1 = pd.DataFrame(data, columns=words)

    f, (ax1) = plt.subplots(figsize=(len(words), 3), nrows=1)
    # sns.heatmap(d1, annot=True, ax=ax1)

    sns.heatmap(d1, annot=True, ax=ax1, linewidths=1, vmax=1, fmt='.2f', vmin=0, center=0.3,
                cmap='Reds')  # YlGnBu,YlOrRd,YlGn,Reds
    plt.show()


def plot_attentions(words, attentions, labels, title):
    """

    :param words:
    :param attention:
    :return:
    """
    datas = []
    data = {}
    columns = []
    max_word_num_per_subplot = 30
    for i in range(len(words)):
        columns.append(words[i])
        data[words[i]] = {}
        for j in range(len(labels)):
            data[words[i]][labels[j]] = attentions[j][i]
        if (i > 0 and i % max_word_num_per_subplot == 0) or i == len(words) - 1:
            d = pd.DataFrame(data, columns=columns)
            datas.append(d)
            data = {}
            columns = []

    nrows = len(datas)
    f, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(80, len(labels) + 5))
    if nrows == 1:
        axes = [axes]
    # sns.heatmap(d1, annot=True, ax=ax1)
    # https://blog.csdn.net/ChenHaoUESTC/article/details/79132602
    axes[0].set_title(title)
    for i, axis in enumerate(axes):
        # axis.legend(fontsize=4)
        label_x = axis.get_xticklabels()
        rotation = 30
        plt.setp(label_x, rotation=rotation)
        sns_plot=sns.heatmap(datas[i], annot=True, ax=axis, linewidths=1, vmax=1, fmt='.2f', vmin=0, center=0.3,
                    cmap='Reds',  annot_kws={'size': 5},cbar=False)  # 图内权重值字体大小调整：annot_kws={'size':9,'weight':'bold', 'color':'blue'}；热力图的颜色风格：YlGnBu,YlOrRd,YlGn,Reds;颜色bar 的设置：1、取消颜色条：cbar=False，2、绘制：cbar_kws={"orientation": "vertical"},
        sns_plot.tick_params(labelsize=5)# X轴和Y轴坐标字体大小
        cb = sns_plot.figure.colorbar(sns_plot.collections[0],shrink=1)  # 单独设置来显示colorbar,缩放比例
        cb.ax.tick_params(labelsize=5)  # 设置colorbar刻度字体大小。
        plt.tight_layout(pad=5)
        plt.subplots_adjust(left=0.03, right=1, top=0.7, bottom=0.35)
    plt.show()


def plot_attentions_pakdd(words, attentions, labels, title):
    """

    :param words:
    :param attention:
    :return:
    """
    datas = []
    data = {}
    columns = []
    max_word_num_per_subplot = 30
    for i in range(len(words)):
        columns.append(words[i])
        data[words[i]] = {}
        for j in range(len(labels)):
            data[words[i]][labels[j]] = attentions[j][i]
        if (i > 0 and i % max_word_num_per_subplot == 0) or i == len(words) - 1:
            d = pd.DataFrame(data, columns=columns)
            datas.append(d)
            data = {}
            columns = []

    nrows = len(datas)
    f, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(80, len(labels) + 5))
    if nrows == 1:
        axes = [axes]
    # sns.heatmap(d1, annot=True, ax=ax1)
    # https://blog.csdn.net/ChenHaoUESTC/article/details/79132602
    axes[0].set_title(title)
    for i, axis in enumerate(axes):
        # axis.legend(fontsize=4)
        # label_x = axis.get_xticklabels()
        # x_rotation = 0
        # plt.setp(label_x, rotation=x_rotation)
        # label_y = axis.get_yticklabels()
        # y_rotation = 190
        # plt.setp(label_y, rotation=y_rotation)
        sns_plot = sns.heatmap(datas[i], annot=True, ax=axis, linewidths=1, vmax=1, fmt='.2f', vmin=0, center=0.3,
                               cmap='Reds', annot_kws={'size': 5}, cbar=True, cbar_kws={
                'shrink': 1})  # YlGnBu,YlOrRd,YlGn,Reds；, 颜色bar 设置：cbar_kws={"orientation": "vertical"}
        # sns_plot.tick_params(labelsize=5)# X轴和Y轴坐标字体大小
        # cb = sns_plot.figure.colorbar(sns_plot.collections[0], shrink=1)  # 单独设置来显示colorbar,缩放比例
        # cb.ax.tick_params(labelsize=5)  # 设置colorbar刻度字体大小。
        # plt.tight_layout(pad=5)
        # plt.subplots_adjust(left=0.03, right=1, top=0.7, bottom=0.35)
    plt.show()


def plot_multi_attentions_of_sentence(words, attentions_list, labels, titles, savefig_filepath=None):
    """

    :param words:
    :param attention:
    :return:
    """
    datas = []
    for i in range(len(titles)):
        data = {}
        columns = range(len(words))
        if i == len(titles) - 1:
            columns = ['%s-%s' % (str(words[k]), str(k)) for k in range(len(words))]
        for j in range(len(words)):
            column_name = columns[j]
            data[column_name] = {labels[i]: attentions_list[i][j]}
        data = pd.DataFrame(data, columns=columns)
        datas.append(data)

    nrows = len(datas)
    f, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(80, len(labels) + 5))
    if nrows == 1:
        axes = [axes]
    # sns.heatmap(d1, annot=True, ax=ax1)
    # https://blog.csdn.net/ChenHaoUESTC/article/details/79132602

    for i, axis in enumerate(axes):
        axis.set_title(titles[i], fontsize=15)
        label_x = axis.get_xticklabels()
        rotation = 30
        plt.setp(label_x, rotation=rotation)
        sns_plot = sns.heatmap(datas[i], annot=True, ax=axis, linewidths=1, vmax=1, fmt='.2f', vmin=0, center=0.3,
                    cmap='Reds', annot_kws={'size': 15}, cbar=True, cbar_kws={'shrink': 1})  # YlGnBu,YlOrRd,YlGn,Reds；, 颜色bar 设置：cbar_kws={"orientation": "vertical"}
        sns_plot.tick_params(labelsize=15)  # X轴和Y轴坐标字体大小
        # cb = sns_plot.figure.colorbar(sns_plot.collections[0], shrink=1)  # 单独设置来显示colorbar,缩放比例
        # cb.ax.tick_params(labelsize=5)  # 设置colorbar刻度字体大小。
        plt.tight_layout(pad=5)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.09)
    if savefig_filepath:
        plt.savefig(savefig_filepath, format='svg')
    else:
        plt.show()


def plot_multi_attentions_of_sentence_backup(words, attentions_list, labels, titles, savefig_filepath=None):
    """

    :param words:
    :param attention:
    :return:
    """
    datas = []
    for i in range(len(titles)):
        data = {}
        columns = range(len(words))
        if i == len(titles) - 1:
            columns = ['%s-%s' % (str(words[k]), str(k)) for k in range(len(words))]
        for j in range(len(words)):
            column_name = columns[j]
            data[column_name] = {labels[i]: attentions_list[i][j]}
        data = pd.DataFrame(data, columns=columns)
        datas.append(data)

    nrows = len(datas)
    f, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(80, len(labels) + 5))
    if nrows == 1:
        axes = [axes]
    # sns.heatmap(d1, annot=True, ax=ax1)
    # https://blog.csdn.net/ChenHaoUESTC/article/details/79132602

    for i, axis in enumerate(axes):
        axis.set_title(titles[i])
        label_x = axis.get_xticklabels()
        rotation = 30
        plt.setp(label_x, rotation=rotation)
        sns_plot = sns.heatmap(datas[i], annot=True, ax=axis, linewidths=1, vmax=1, fmt='.2f', vmin=0, center=0.3,
                    cmap='Reds', annot_kws={'size': 5}, cbar=True, cbar_kws={'shrink': 1})  # YlGnBu,YlOrRd,YlGn,Reds；, 颜色bar 设置：cbar_kws={"orientation": "vertical"}
        sns_plot.tick_params(labelsize=5)  # X轴和Y轴坐标字体大小
        # cb = sns_plot.figure.colorbar(sns_plot.collections[0], shrink=1)  # 单独设置来显示colorbar,缩放比例
        # cb.ax.tick_params(labelsize=5)  # 设置colorbar刻度字体大小。
        plt.tight_layout(pad=5)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.09)
    if savefig_filepath:
        plt.savefig(savefig_filepath, format='svg')
    else:
        plt.show()


def extract_numbers(text: str):
    """
    I(0.00) go(0.00) -> 0.00 0.00
    :param text:
    :return:
    """
    nums = re.findall('[0-9]\.[0-9]+', text)
    result = [float(num) for num in nums]
    return result


if __name__ == '__main__':
    # plot_attention()
    # plot_multi_attentions_of_sentence()

    words = 'I go to Sushi Rose for fresh sushi and great portions all at a reasonable price'.split()
    attentions = [
        "I(0.00) go(0.00) to(0.00) Sushi(0.06) Rose(0.00) for(0.00) fresh(0.07) sushi(0.63) and(0.01) great(0.00) portions(0.22) all(0.01) at(0.00) a(0.00) reasonable(0.00) price(0.00)",
        "I(0.01) go(0.00) to(0.00) Sushi(0.23) Rose(0.00) for(0.00) fresh(0.22) sushi(0.23) and(0.00) great(0.00) portions(0.23) all(0.01) at(0.00) a(0.00) reasonable(0.01) price(0.00)"]
    attentions = [extract_numbers(attention) for attention in attentions]
    labels = ['lstm', 'affine']
    # plot_attentions(words, attentions, labels, 'test_plot_attentions')
    titles = ['t1', 't2']
    plot_multi_attentions_of_sentence(words, attentions, labels, ['t1', 't2'])
    plot_multi_attentions_of_sentence(words, attentions, labels, titles)
    # plot_attentions_pakdd(words, attentions + attentions, labels + labels, 'comparision')