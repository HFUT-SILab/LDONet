import os


def config(dataset_path):
    if dataset_path.split("/")[-1] == "PolyUII_20":
        train_class, test_start_class, test_end_class = 193, 194, 386
    elif dataset_path.split("/")[-1] == "Blue":
        train_class, test_start_class, test_end_class = 250, 251, 500
    elif dataset_path.split("/")[-1] == "Green":
        train_class, test_start_class, test_end_class = 250, 251, 500
    elif dataset_path.split("/")[-1] == "NIR":
        train_class, test_start_class, test_end_class = 250, 251, 500
    elif dataset_path.split("/")[-1] == "Red":
        train_class, test_start_class, test_end_class = 250, 251, 500
    elif dataset_path.split("/")[-1] == "HFUT":
        train_class, test_start_class, test_end_class = 400, 401, 800
    elif dataset_path.split("/")[-1] == "CS":
        train_class, test_start_class, test_end_class = 100, 101, 200
    elif dataset_path.split("/")[-1] == "TJC":
        train_class, test_start_class, test_end_class = 300, 301, 600
    return train_class, test_start_class, test_end_class


def custom_sort(filename):
    # 按照a为第一排序，b为第二排序进行升序排序
    a, b = filename.split('_')
    return (int(a), int(b.split('.')[0]))


def generate_open_set_split(sorted_files, dataset_path, dataset_name, root):
    train_class, test_start_class, test_end_class = config(dataset_name)

    with open(os.path.join(root, f'train_{dataset_name}.txt'), 'w') as ofs:
        for filename in sorted_files:
            a, _ = filename.split('_')
            a = int(a)
            if a <= train_class:
                imagePath = os.path.join(dataset_path, filename)
                userID = int(a) - 1
                ofs.write('%s %d\n' % (imagePath, userID))

    with open(os.path.join(root, f'test_{dataset_name}.txt'), 'w') as ofs:
        for filename in sorted_files:
            a, _ = filename.split('_')
            a = int(a)
            if test_start_class <= a <= test_end_class:
                imagePath = os.path.join(dataset_path, filename)
                userID = int(a) - 1
                ofs.write('%s %d\n' % (imagePath, userID))


def generate_closed_set_split(sorted_files, dataset_path, dataset_name, root):
    # 闭集划分：同一类中前一半样本用于训练，后一半样本用于测试
    class_to_files = {}
    for filename in sorted_files:
        a, _ = filename.split('_')
        class_id = int(a)
        class_to_files.setdefault(class_id, []).append(filename)

    train_path = os.path.join(root, f'train_{dataset_name}_closed.txt')
    test_path = os.path.join(root, f'test_{dataset_name}_closed.txt')

    with open(train_path, 'w') as train_ofs, open(test_path, 'w') as test_ofs:
        for class_id in sorted(class_to_files.keys()):
            class_files = class_to_files[class_id]
            split_idx = len(class_files) // 2

            train_files = class_files[:split_idx]
            test_files = class_files[split_idx:]

            userID = class_id - 1

            for filename in train_files:
                imagePath = os.path.join(dataset_path, filename)
                train_ofs.write('%s %d\n' % (imagePath, userID))

            for filename in test_files:
                imagePath = os.path.join(dataset_path, filename)
                test_ofs.write('%s %d\n' % (imagePath, userID))


root = './'
path1 = r"D:\palm_datasets\Palmprint\PolyUM\Green"
dataset_name = path1.split('\\')[-1]
files = os.listdir(path1)
sorted_files = sorted(files, key=custom_sort)
# 保留原开集划分
#generate_open_set_split(sorted_files, path1, dataset_name, root)

# 新增闭集划分
generate_closed_set_split(sorted_files, path1, dataset_name, root)

