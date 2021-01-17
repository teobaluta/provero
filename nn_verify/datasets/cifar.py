"""CIFAR dataset input module.
"""

import tensorflow as tf
import os, sys
import tarfile
from six.moves import urllib

def build_tests(dataset, data_path, batch_size, standardize_images):
    """
    Simplify the dataset loading
    """
    image_size = 32
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0
        num_classes = 10
        data_path = maybe_download_and_extract('10', data_path)
    elif dataset == 'cifar100':
        label_bytes = 1
        label_offset = 1
        num_classes = 100
        data_path = maybe_download_and_extract('100', data_path)
    else:
        raise ValueError('Not supported dataset %s', dataset)

    if dataset == 'cifar10':
        data_path = os.path.join(data_path, 'test_batch*')
    elif dataset == 'cifar100':
        data_path = os.path.join(data_path, 'test.bin')
    else:
        raise ValueError('Dataset %s does not exist', dataset)

    depth = 3
    image_bytes = image_size * image_size * depth
    record_bytes = label_bytes + label_offset + image_bytes

    data_files = tf.gfile.Glob(data_path)
    file_queue = tf.train.string_input_producer(data_files, shuffle=False)
    # Read examples from files in the filename queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)

    # Convert these examples to dense labels and processed images.
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
    # Convert from string to [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record, [label_offset + label_bytes], [image_bytes]),
                             [depth, image_size, image_size])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    image = tf.image.resize_image_with_crop_or_pad(
        image, image_size, image_size)
    if standardize_images:
        image = tf.image.per_image_standardization(image)
    else:
        image = image / 255.0 - 0.5

    example_queue = tf.FIFOQueue(
        3 * batch_size,
        dtypes=[tf.float32, tf.int32],
        shapes=[[image_size, image_size, depth], [1]])
    num_threads = 1

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images_from_queue, labels_from_queue = example_queue.dequeue_many(batch_size)
    images = tf.compat.v1.placeholder_with_default(images_from_queue,
                                                   shape=(batch_size, image_size,
                                                         image_size, depth))
    labels_from_queue = tf.reshape(labels_from_queue, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels_from_queue = tf.sparse_to_dense(
        tf.concat(values=[indices, labels_from_queue], axis=1),
        [batch_size, num_classes], 1.0, 0.0)
    labels = tf.compat.v1.placeholder_with_default(labels_from_queue,
                                                   shape=(batch_size,
                                                          num_classes))

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    return images, labels

def build_input(dataset, data_path, batch_size, standardize_images, mode):
    """Build CIFAR image and labels.

    Args:
      dataset: Either 'cifar10' or 'cifar100'.
      data_path: Filename for data.
      batch_size: Input batch size.
      mode: Either 'train' or 'eval'.
    Returns:
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
    Raises:
      ValueError: when the specified dataset is not supported.
    """
    image_size = 32
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0
        num_classes = 10
        data_path = maybe_download_and_extract('10', data_path)
    elif dataset == 'cifar100':
        label_bytes = 1
        label_offset = 1
        num_classes = 100
        data_path = maybe_download_and_extract('100', data_path)
    else:
        raise ValueError('Not supported dataset %s', dataset)

    if mode == 'train' and dataset == 'cifar10':
        data_path = os.path.join(data_path, 'data_batch*')
    elif mode == 'train' and dataset == 'cifar100':
        data_path = os.path.join(data_path, 'train.bin')
    elif mode == 'eval' and dataset == 'cifar10':
        data_path = os.path.join(data_path, 'test_batch*')
    elif mode == 'eval' and dataset == 'cifar100':
        data_path = os.path.join(data_path, 'test.bin')
    else:
        raise ValueError('Mode %s does not exist', mode)

    depth = 3
    image_bytes = image_size * image_size * depth
    record_bytes = label_bytes + label_offset + image_bytes

    data_files = tf.gfile.Glob(data_path)
    file_queue = tf.train.string_input_producer(data_files, shuffle=(mode == 'train'))
    # Read examples from files in the filename queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)

    # Convert these examples to dense labels and processed images.
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
    # Convert from string to [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record, [label_offset + label_bytes], [image_bytes]),
                             [depth, image_size, image_size])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    if mode == 'train':
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size+4, image_size+4)
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)

        if standardize_images:
            image = tf.image.per_image_standardization(image)
        else:
            image = image / 255.0 - 0.5

        example_queue = tf.RandomShuffleQueue(
            capacity=16 * batch_size,
            min_after_dequeue=8 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 16
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size, image_size)
        if standardize_images:
            image = tf.image.per_image_standardization(image)
        else:
            image = image / 255.0 - 0.5

        example_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 1

    print('mode={}, {}'.format(mode, image))
    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, num_classes], 1.0, 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    return images, labels

def maybe_download_and_extract(class_n, dest_directory):
    """Download and extract the tarball.

    Args:
      dataset: '10' or '100' for respectice cifar dataset.
      dest_directory: Dirpath for all data.
    Returns:
      images: Dirpath with cifar data.
    """

    data_url = 'https://www.cs.toronto.edu/~kriz/cifar-{}-binary.tar.gz'.format(class_n)

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    if class_n == '10':
        extracted_dir_path = os.path.join(
            dest_directory,
            'cifar-{}-batches-bin'.format(class_n)
        )
    elif class_n == '100':
        extracted_dir_path = os.path.join(
            dest_directory,
            'cifar-{}-binary'.format(class_n)
        )

    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    return extracted_dir_path

