import numpy as np

def progress_bar(iter, total, prefix = '', suffix = '', length=100, fill = '█', print_end='\r'):
    percent = ('{0:.2f}'.format(100 * (iter / total)))
    filled_length = int(length * iter // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}%', end='\n' if iter == total else print_end)

def hot_encode(Y, classes=None):
    min_Y, max_Y = int(np.min(Y)), int(np.max(Y))
    if classes == None:
        classes = max_Y - min_Y + 1
    dict = {}
    for i in range(min_Y, classes + min_Y):
        v = np.zeros((classes, ))
        v[i-min_Y] = 1
        dict[i] = v
    return np.array([dict[y] for y in Y])

def batch_normalization(X):
    """
    X should have shape (n, m), where n is number of features and m the number of examples.
    """
    for x in X:
        x -= np.mean(x)
        x /= np.sqrt(np.sum(x**2) / X.shape[1])
    return X

# file_handler, X, Y = open('ex1data2.txt'), [], []
# for m, line in enumerate(file_handler):
#     values = line.rstrip().split(',')
#     X.append([float(v) for v in values[:-1]])
#     Y.append(float(values[-1]))

# X = np.array(X).reshape(m + 1, -1).T
# Y = np.array(Y).reshape(m + 1, -1).T

# batch_normalization(X)