import gzip
import numpy as np

class mnist(object):
    def __init__(self):
      self.epostart=0
      self.epocomp=0
      self.index_in_epoch=0
      self.TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
      self.TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
      self.TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
      self.TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    def _read32(self,bytestream):
      dt = np.dtype(np.uint32).newbyteorder('>')
      return np.frombuffer(bytestream.read(4), dtype=dt)[0]
    def extract_images(self,file):
      with gzip.GzipFile(fileobj=file) as bytestream:
         d=self._read32(bytestream)    
         num = self._read32(bytestream)
         rows = self._read32(bytestream)
         cols = self._read32(bytestream)
         buf = bytestream.read(rows * cols * num)
         data = np.frombuffer(buf, dtype=np.uint8)
         data = data.reshape(num, rows, cols, 1)
         return data

    def dense_to_one_hot(self,labels, num_classes):
      
       num_labels = labels.shape[0]
       index_offset = np.arange(num_labels) * num_classes
       labels_one_hot = np.zeros((num_labels, num_classes))
       labels_one_hot.flat[index_offset + labels.ravel()] = 1
       return labels_one_hot

    def extract_labels(self,file):
      with gzip.GzipFile(fileobj=file) as bytestream:
         d=self._read32(bytestream)    
         num = self._read32(bytestream)
         buf = bytestream.read(num)
         labels= np.frombuffer(buf, dtype=np.uint8)
         return self.dense_to_one_hot(labels,10)
    def load(self):
        with open(self.TRAIN_IMAGES, 'rb') as file:
         #loading and processing train_images
         train_images = self.extract_images(file)
         train_images = train_images.reshape(train_images.shape[0],train_images.shape[1] * train_images.shape[2])
         train_images = train_images.astype(np.float32)
         train_images = np.multiply(train_images, 1.0 / 255.0)
         
        with open(self.TRAIN_LABELS, 'rb') as file:
         #loading train_labels
         train_labels = self.extract_labels(file)
         

        with open(self.TEST_IMAGES, 'rb') as file:
         #loading and processing train_images 
         test_images = self.extract_images(file)
         test_images = test_images.reshape(test_images.shape[0],test_images.shape[1] * test_images.shape[2])
         test_images = test_images.astype(np.float32)
         test_images = np.multiply(test_images, 1.0 / 255.0)
         
        with open(self.TEST_LABELS, 'rb') as file:
         #loading test labels
         test_labels = self.extract_labels(file)
        return train_images,train_labels,test_images,test_labels
    def next_batch(self,batch_size,test=False):
        #loading data
        images,labels,_images,_labels=self.load()
        if test:
            #return test data
            return _images[0:batch_size],_labels[0:batch_size]
        start = self.epostart
        num=images.shape[0] #number of samples
        
        if start + batch_size > num:
           #when all samples are visited
           #looping again
           rest_num = num - start
           images_rest = images[start:num]
           labels_rest = labels[start:num]
           
           start = 0
           self.index_in_epoch = batch_size - rest_num
           
           end = self.index_in_epoch
           images_new_part = images[start:end]
           labels_new_part = labels[start:end]
           #Shuffling images
           perm=images_rest.shape[0]+images_new_part.shape[0]
           perm = np.arange(perm)
           np.random.shuffle(perm)
           
           images=np.concatenate((images_rest, images_new_part), axis=0)
           labels=np.concatenate((labels_rest, labels_new_part), axis=0)
           
           
           return images[perm],labels[perm]
        else:
           self.index_in_epoch += batch_size
           end = self.index_in_epoch
           self.epostart=end
           images=images[start:end] 
           labels=labels[start:end]
           data=[images,labels]
           perm0 = np.arange(batch_size)
           np.random.shuffle(perm0)
           images = images[perm0]
           labels = labels[perm0]
           
           
           return images,labels
 




