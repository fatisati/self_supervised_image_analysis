import matplotlib.pyplot as plt


def type1():
    fig = plt.figure()
    rows, cols = 2, 3
    cnt = 1
    for i in range(rows * cols):
        fig.add_subplot(rows, cols, cnt)
        cnt += 1


def multiple_plots(x_arr, y_arr, titles, rows, cols):
    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            axs[row, col].plot(x_arr[idx], y_arr[idx])
            axs[row, col].set_title(titles[idx])
            idx += 1
            if idx == len(x_arr):
                return
