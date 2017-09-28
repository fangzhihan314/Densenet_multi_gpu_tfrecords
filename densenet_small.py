import numpy as np
import tensorflow as tf
from tensorflow import logging
from tensorflow.python.client import device_lib
from tensorflow import flags
import os
import pdb

logging.set_verbosity(tf.logging.INFO)

FLAGS = flags.FLAGS

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
        'label':tf.FixedLenFeature([],tf.int64),
        'image_raw':tf.FixedLenFeature([],tf.string)
        })
    image = tf.decode_raw(features['image_raw'],tf.uint8)
    label_ = tf.cast(features['label'],tf.int32)
    label = tf.one_hot(label_, FLAGS.num_label)
    IMG_PIXELS = FLAGS.image_size * FLAGS.image_size * FLAGS.image_channel
    image.set_shape([IMG_PIXELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    return image,label

def inputs(data_set, base_dir):
    files = os.listdir(base_dir)
    file_ = []
    if data_set == 'train':
        for f in files:
            if f.startswith('train'):
                tmp = "%s/%s" % (base_dir, f)
                file_.append(tmp)
    else:
        for f in files:
            if f.startswith('test'):
                tmp = "%s/%s" % (base_dir, f)
                file_.append(tmp)

    filename_queue = tf.train.string_input_producer(file_, num_epochs = FLAGS.num_epoch)
    return filename_queue

    # return images,labels

def get_total_num_train(data_set, base_dir):
  files = os.listdir(base_dir)
  file_ = []
  if data_set == 'train':
      for f in files:
          if f.startswith('train'):
              tmp = "%s/%s" % (base_dir, f)
              file_.append(tmp)
  else:
      for f in files:
          if f.startswith('test'):
              tmp = "%s/%s" % (base_dir, f)
              file_.append(tmp)

  num_total_files = 0
  for fn in file_:
    num_total_files += sum(1 for _ in tf.python_io.tf_record_iterator(fn))
  return num_total_files

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      # tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      # tf.summary.scalar('max', tf.reduce_max(var))
      # tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


def run_in_batch_avg(session, tensors, batch_placeholders, merged, test_writer, 
  summary_count, feed_dict={}, batch_size=200):
  test_count = summary_count
  res = [ 0 ] * len(tensors)                                                                                           
  batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]                    
  total_size = len(batch_tensors[0][1]) 

  # batch_count = (total_size + batch_size - 1) / batch_size  
  batch_count = (total_size) / batch_size  # due to the multiple GPU 
  total_size = batch_count * batch_size
  tensors.append(merged)
                                      
  for batch_idx in xrange(batch_count):                                                                                
    current_batch_size = None                                                                                          
    for (placeholder, tensor) in batch_tensors:                                                                        
      batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                                         
      current_batch_size = len(batch_tensor)                                                                           
      feed_dict[placeholder] = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                               
    
    tmp = session.run(tensors, feed_dict=feed_dict)
    test_writer.add_summary(tmp[-1], test_count)
    test_count += 1                                                                  
    res = [ r + t * current_batch_size for (r, t) in zip(res, tmp[:-1]) ]                                                  
  return [ r / float(total_size) for r in res ]

def conv2d(input, in_features, out_features, kernel_size, id, with_bias=False, stride = 1):
  shape = [ kernel_size, kernel_size, in_features, out_features ]
  w_name = "weight_conv_%d" % id
  W = tf.get_variable(w_name, shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
  conv = tf.nn.conv2d(input, W, [ 1, stride, stride, 1 ], padding='SAME')
  if with_bias:
    b_name = "bias_conv_%d" % id
    tmp = tf.get_variable(b_name, [ out_features ], initializer=tf.constant_initializer(0.01))
    return conv + tmp
  return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob, id):
  # "id" is used to differ multiple w and b in function conv2d (whj)
  current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  current = tf.nn.relu(current)
  current = conv2d(current, in_features, out_features, kernel_size, id)
  current = tf.nn.dropout(current, keep_prob)
  return current

def block(input, layers, in_features, growth, is_training, keep_prob, id):
  current = input
  features = in_features
  for idx in xrange(layers):
    new_id = id*10 + idx
    # This is added by whj according to the paper
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob, new_id)
    new_id = id*100 + idx
    tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob, new_id)
    current = tf.concat((current, tmp), 3)
    features += growth
  return current, features

def avg_pool(input, ksize, stride, padding):
  return tf.nn.avg_pool(input, [ 1, ksize, ksize, 1 ], [1, stride, stride, 1 ], padding)

def inference(xs, ys, keep_prob, is_training, label_count, scope):
  weight_decay = 1e-4

  with tf.variable_scope('input_reshape'):
    current = tf.reshape(xs, [ -1, FLAGS.image_size, FLAGS.image_size, 3 ])
    tf.summary.image('input', current, 10)
  with tf.variable_scope('conv2d'):
    # current = conv2d(current, 3, 16, 3, 0) # the last one is the id for variable scope (whj)
    current = conv2d(current, 3, 16, 7, 0, stride = 2)
    current = avg_pool(current, 3, 2, 'SAME')
  with tf.variable_scope('block_1'):
    current, features = block(current, 6, 16, FLAGS.growth, is_training, keep_prob, 1)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob, 2)
    current = avg_pool(current, 2, 2, 'VALID')
  with tf.variable_scope('block_2'):
    current, features = block(current, 12, features, FLAGS.growth, is_training, keep_prob, 3)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob, 4)
    current = avg_pool(current, 2,2, 'VALID')
  with tf.variable_scope('block_3'):
    current, features = block(current, 12, features, FLAGS.growth, is_training, keep_prob, 5)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob, 6)
    current = avg_pool(current, 2,2, 'VALID')
  with tf.variable_scope('block_4'):
    current, features = block(current, 6, features, FLAGS.growth, is_training, keep_prob, 7)
  with tf.variable_scope('rest_layers'):
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7, 'VALID') 
    final_dim = features
    current = tf.reshape(current, [ -1, final_dim ])
    with tf.name_scope('weights'):
      Wfc = tf.get_variable("weight_rest", [ final_dim, label_count ], 
        initializer=tf.truncated_normal_initializer(stddev=0.01))
      variable_summaries(Wfc)
    with tf.name_scope('biases'):
      bfc = tf.get_variable("bias_rest", [ label_count ], 
        initializer=tf.constant_initializer(0.01))
      variable_summaries(bfc)
    ys_ = tf.nn.softmax( tf.matmul(current, Wfc) + bfc )
  with tf.variable_scope('loss'):
    cross_entropy = -tf.reduce_mean(ys * tf.log(ys_ + 1e-12))
    tf.summary.scalar('cross_entropy', cross_entropy)
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    loss = cross_entropy + l2 * weight_decay

  correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  tf.summary.scalar('accuracy', accuracy)

  return loss, cross_entropy, accuracy

def run_model(image_dim, label_count):
  # Get the total number of training data
  num_total_train = get_total_num_train('train', FLAGS.data_dir)
  num_total_test = get_total_num_train('test', FLAGS.data_dir)
  graph = tf.Graph()

  # Choose gpu or cpu
  local_device_protos = device_lib.list_local_devices()
  gpus = [x.name for x in local_device_protos if x.device_type == 'GPU'] 
  # gpus = [local_device_protos[1].name]
  num_gpus = len(gpus)

  if num_gpus > 0:
    logging.info("Using the following GPUs to train: " + str(gpus))
    num_towers = num_gpus
    device_string = '/gpu:%d'
  else:
    logging.info("No GPUs found. Training on CPU.")
    num_towers = 1
    device_string = '/cpu:%d'


  # Build the graph
  with graph.as_default(), tf.device('/cpu:0'):
    batch_size = FLAGS.batch_size * num_towers
    lr = tf.placeholder("float", shape=[])
    keep_prob = tf.placeholder(tf.float32)
    # is_training = tf.placeholder("bool", shape=[])
    is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

    # The input data
    train_filename_queue = inputs('train', FLAGS.data_dir)
    # test_filename_queue = inputs('test', FLAGS.data_dir)
    # # q_selector = tf.placeholder(tf.int32, [])  # 0 is for training and 1 is for testing
    # q_selector = tf.cond(is_training, lambda: tf.constant(0), lambda: tf.constant(1))
    # q = tf.QueueBase.from_list(q_selector, [train_filename_queue, test_filename_queue])

    image,label = read_and_decode(train_filename_queue)
    xs,ys = tf.train.shuffle_batch([image, label], 
        batch_size=batch_size,
        num_threads=2,
        capacity=100 + 3 * batch_size,
        min_after_dequeue=100
    )
    
    global_step = tf.get_variable(
      'global_step', [], initializer=tf.constant_initializer(0), 
       trainable=False)

    tower_cross_entropy = []
    tower_accuracy = []
    tower_gradients = []
    tower_xs = tf.split(xs, num_towers)
    tower_ys = tf.split(ys, num_towers)

    opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
    # opt = tf.train.GradientDescentOptimizer(lr)

    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(num_towers):
        with tf.device(device_string % i):
          with tf.name_scope('tower_%d' %  i) as scope:
            loss, cross_entropy, accuracy = inference(tower_xs[i], tower_ys[i],
              keep_prob, is_training, label_count, scope)

            tf.get_variable_scope().reuse_variables()

            gradients = opt.compute_gradients(loss)
            
            tower_cross_entropy.append(cross_entropy)
            tower_accuracy.append(accuracy)
            tower_gradients.append(gradients)

    cross_entropy = tf.reduce_mean(tf.stack(tower_cross_entropy))
    tf.summary.scalar('cross_entropy_ave', cross_entropy)
    accuracy = tf.reduce_mean(tf.stack(tower_accuracy))
    tf.summary.scalar('accuracy_ave', accuracy)

    merged_gradients = average_gradients(tower_gradients)
    apply_gradient_op = opt.apply_gradients(
          merged_gradients, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(
        tf.trainable_variables())
    train_op = tf.group(apply_gradient_op, variables_averages_op)

  # RUN
  config = tf.ConfigProto(allow_soft_placement = True)  # For the tf error of using last gpu
  with tf.Session(graph=graph, config = config) as session:
    merged = tf.summary.merge_all()
    
    # The dirs of summaries
    log_train = "%s/train/" % FLAGS.log_dir
    log_test = "%s/test/" % FLAGS.log_dir
    train_writer = tf.summary.FileWriter(log_train, session.graph)
    test_writer = tf.summary.FileWriter(log_test)

    learning_rate = 0.1
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord) 
    saver = tf.train.Saver(tf.global_variables())

    step_count = 0
    try:
      while not coord.should_stop():

        current_count = FLAGS.batch_size * num_towers * step_count
        pro_rate = (float)(current_count) / (float)(num_total_train * FLAGS.num_epoch)
        current_epoch = current_count / num_total_train
        if pro_rate > 0.5 and pro_rate < 0.75:
          learning_rate = 0.01
        if pro_rate >= 0.75:
          learning_rate = 0.001

        batch_res = session.run([merged, train_op, cross_entropy, accuracy ],
          feed_dict = {lr: learning_rate, is_training: True, keep_prob: 0.8 })
        train_writer.add_summary(batch_res[0], step_count)
        step_count += 1
        logging.info("epoch %d, step %d, cross_entropy %f, acc %f" % 
          (current_epoch, step_count, batch_res[2], batch_res[3]))
        if step_count % 100 == 0:
          save_path = saver.save(session, 'logs/densenet_step%06d.ckpt' % step_count)

        # test_results = run_in_batch_avg(session, [ cross_entropy, accuracy ], [ xs, ys ], 
        #   merged, test_writer, summary_count, 
        #   feed_dict = { xs: data['test_data'], ys: data['test_labels'], is_training: False, keep_prob: 1. }, 
        #   batch_size = 200*num_towers)
    except tf.errors.OutOfRangeError:
      logging.info('Doneb training -- epoch limit reached' )
    finally:
      # When donw, ask the threads to stop
      coord.request_stop()
      coord.join(threads)

    session.close()
    train_writer.close()
    test_writer.close()


      # test_results = run_in_batch_avg(session, [ cross_entropy, accuracy ], [ xs, ys ],  
      #   merged, test_writer, summary_count, 
      #   feed_dict = { xs: data['test_data'], ys: data['test_labels'], is_training: False, keep_prob: 1. }, 
      #   batch_size = 200*num_towers)
      # logging.info("Test result: cross_entropy %f, acc %f" % (test_results[0], test_results[1]))

def run():
  image_size = FLAGS.image_size
  image_dim = image_size * image_size * 3
  label_count = FLAGS.num_label 

  run_model(image_dim, label_count)

if __name__ == '__main__':
  # flags.DEFINE_string("data_dir", "../../../../data/flower_photos/tfrecords",
  #                     "The folder where data is")
  flags.DEFINE_string("data_dir", "/mnt/yardcephfs/mmyard/g_wxg_td_prc/img/hankinwang/densenet/flower_data",
                      "The folder where data is")

  flags.DEFINE_string("log_dir", "logs",
                      "The folder where the log will be")
  flags.DEFINE_integer("image_size", 224,
                      "The size of the input image")
  flags.DEFINE_integer("image_channel", 3,
                      "The channel of the input image")
  flags.DEFINE_integer("batch_size", 4,    # 4
                      "rt")
  flags.DEFINE_integer("num_label", 5,
                      "The number of labels")
  flags.DEFINE_integer("num_epoch", 500,
                      "the number of epoch")
  flags.DEFINE_integer("growth", 6,
                      "The growth in the densenet, i.e., k")
  flags.DEFINE_float("moving_average_decay", 0.99,
                      "MOVING_AVERAGE_DECAY")

  # Run
  run()
