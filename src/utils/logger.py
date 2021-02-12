# IMPORT LIBRARIES
import tensorflow as tf
import numpy as np
from io import BytesIO
import os
from PIL import Image
class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(log_dir)
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
        

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            
            byte_io = BytesIO()
            #im = Image.fromarray((x * 255).astype(np.uint8))
            img = np.transpose(img,(1,2,0))
            #img = img.reshape(32, 32)
            im = Image.fromarray((img*255).astype(np.uint8))
            im.save(byte_io, format="png")
            
            # Create an Image object
            image = tf.image.decode_png(byte_io.getvalue(), channels=1)
            image = tf.expand_dims(image, 0)
            
            
            with self.writer.as_default():
                imag = tf.summary.image("Training data", image, step=0)
                
                img_sum = tf.summary.scalar('%s/%d' % (tag, i), imag, step)
                img_summaries.append(img_sum)
                
                self.writer.flush()
            

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.summary(value=[tf.summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()