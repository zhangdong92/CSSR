import os


def create_parent_dir(path):
    index=path.rfind("\\")
    index2=path.rfind("/")

    index=index if index>index2 else index2
    parent_path=path[:index]
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
        print("makedirs path ={}".format(parent_path))

