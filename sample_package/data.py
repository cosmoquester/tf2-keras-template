import tensorflow as tf


def get_dataset(dataset_file_path: str):
    """
    Read dataset file and construct tensorflow dataset

    :param dataset_file_path: dataset file path.
    """
    # Replace the context of this function
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    return dataset
