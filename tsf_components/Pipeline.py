import tensorflow as tf
import tempfile


# Two way to create dataset
# A data transformation constructs a dataset from one or more tf.data.Dataset objects.
datatype = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]],
                                             values=[1, 2],
                                             dense_shape=[3, 4]))

# A data source constructs a Dataset from data stored in memory or in one or more files.
_, filename = tempfile.mkdtemp()
with open(filename, 'w') as f:
    f.write("""Line 1
    Line 2
    Line 3
      """)
ds_file = tf.data.TextLineDataset(filename)

# Spec allows you to inspect the type of each element component
datatype.element_spec

