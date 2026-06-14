import numpy as np


def create_resizable_dataset(handle, name, dtype):
    return handle.create_dataset(
        name,
        shape=(0,),
        maxshape=(None,),
        chunks=True,
        dtype=dtype,
        compression="gzip",
        compression_opts=1,
    )


def append_dataset(dataset, values):
    values = np.asarray(values, dtype=dataset.dtype)
    old_size = dataset.shape[0]
    new_size = old_size + len(values)
    dataset.resize((new_size,))
    dataset[old_size:new_size] = values
