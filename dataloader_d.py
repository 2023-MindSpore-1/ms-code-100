import numpy as np 
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.GeneratorDataset as gd
import os

# 读取数据
def read_flow_data(data_path):
    flow_data = np.load(data_path, allow_pickle=True).astype(np.float32)
    return flow_data

# 划分训练集 验证集 测试集
def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

# 分割X Y
def get_X_Y(cmp_or_sr, uncmp_data, step):
    length = len(cmp_or_sr)
    end_index = length - step
    X, Y = [], []
    index = 0
    while index <= end_index:
        X.append(uncmp_data[index:index + step])
        Y.append(cmp_or_sr[index + step - 1])
        index += 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def data_loader(X, Y, E, batch_size, shuffle=True, drop_last=True):
    X = mindspore.Tensor.from_numpy(X)
    Y = mindspore.Tensor.from_numpy(Y)
    # 如果 external feature 存在
    if E is not None:
        E = mindspore.Tensor.from_numpy(E)
        data = gd(X, Y, E)
    else:
        data = gd(X, Y)

    sampler = ds.SequentialSampler()
    dataset = ds.NumpySlicesDataset(data, sampler=sampler)
    dataset = dataset.batch(batch_size=batch_size)
    dataloader = dataset.create_dict_iterator()

    return dataloader

# five uncompleted data, the fifth completed data
def get_dataloader(batch_size, step, val_ratio, test_ratio, cmp_or_sr_path, uncmp_path, feature_path):
    data = read_flow_data(cmp_or_sr_path)
    uncmp_data = read_flow_data(uncmp_path)

    data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)
    uncmp_data_train, uncmp_data_val, uncmp_data_test = split_data_by_ratio(uncmp_data, val_ratio, test_ratio)

    x_train, y_train = get_X_Y(data_train, uncmp_data_train, step)
    x_val, y_val = get_X_Y(data_val, uncmp_data_val, step)
    x_test, y_test = get_X_Y(data_test, uncmp_data_test, step)

    # 判断feature是否存在
    if os.path.exists(feature_path):   
        feature = read_flow_data(feature_path)
        fea_train, fea_val, fea_test = split_data_by_ratio(feature, val_ratio, test_ratio)
        fea_train, fea_val, fea_test = fea_train[step-1:], fea_val[step-1:], fea_test[step-1:]
    else:
        fea_train, fea_val, fea_test = None, None, None

    train_dataloader = data_loader(x_train, y_train, fea_train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_dataloader = data_loader(x_val, y_val, fea_val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, fea_test, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader



