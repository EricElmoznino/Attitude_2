import tensorflow as tf
import tensorflow.contrib.data as data
from tensorflow.contrib.tensorboard.plugins import projector
import Helpers as hp
import numpy as np
import shutil
import os
import time
from functools import reduce
import operator


class Model:
    def __init__(self, configuration, image_width, image_height):
        self.conf = configuration

        self.image_shape = [image_width, image_height, 3]
        self.label_shape = [3]

        with tf.variable_scope('hyperparameters'):
            self.keep_prob_placeholder = tf.placeholder(tf.float32, name='dropout_keep_probability')

        self.dataset_placeholders, self.datasets, self.iterator = self.create_input_pipeline()
        self.images_ref, self.images_new, self.labels = self.iterator.get_next()
        self.model = self.build_model_two_channel_deep()
        self.saver = tf.train.Saver()

    def create_input_pipeline(self):
        with tf.variable_scope('input_pipeline'):
            images_ref = tf.placeholder(tf.string, [None])
            images_new = tf.placeholder(tf.string, [None])
            labels = tf.placeholder(tf.float32, [None] + self.label_shape)
            placeholders = {'images_ref': images_ref, 'images_new': images_new, 'labels': labels}

            def process_images(img_file_ref, img_file_new, label):
                img_content_ref = tf.read_file(img_file_ref)
                img_ref = tf.image.decode_jpeg(img_content_ref, channels=self.image_shape[-1])
                img_ref = tf.divide(tf.cast(img_ref, tf.float32), 255)
                img_ref.set_shape(self.image_shape)
                img_content_new = tf.read_file(img_file_new)
                img_new = tf.image.decode_jpeg(img_content_new, channels=self.image_shape[-1])
                img_new = tf.divide(tf.cast(img_new, tf.float32), 255)
                img_new.set_shape(self.image_shape)
                return img_ref, img_new, label

            dataset = data.Dataset.from_tensor_slices((images_ref, images_new, labels))
            dataset = dataset.map(process_images)
            dataset = dataset.repeat()
            train_set = dataset.batch(self.conf.batch_size)
            predict_set = dataset.batch(1)
            datasets = {'train': train_set, 'predict': predict_set}

            iterator = data.Iterator.from_dataset(train_set)

        return placeholders, datasets, iterator

    def build_model_two_channel_deep(self):
        filter_sizes = [[4, 4], [4, 4], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        channel_sizes = [40, 80, 80, 160, 160, 320, 320, 640]
        pools = [True, False, True, False, True, False, False, False]

        fully_connected_sizes = [1024, 1024]

        roll_sizes = [1024, 512]
        yaw_pitch_sizes = [1024]
        yaw_sizes = [512]
        pitch_sizes = [512]

        with tf.variable_scope('model'):

            with tf.variable_scope('convolution'):
                conv = tf.concat([self.images_ref, self.images_new], axis=3)
                for i, (filter_size, channel_size, pool) in enumerate(zip(filter_sizes, channel_sizes, pools)):
                    with tf.variable_scope('layer_' + str(i)):
                        conv = hp.convolve(conv, filter_size, int(conv.shape[-1]), channel_size, pad=True)
                        conv = tf.nn.relu(conv)
                        if pool: conv = hp.max_pool(conv, [2, 2], pad=True)

            with tf.variable_scope('fully_connected'):
                input_size = int(conv.shape[1] * conv.shape[2] * conv.shape[3])
                layer = tf.reshape(conv, [-1, input_size])
                for i, layer_size in enumerate(fully_connected_sizes):
                    with tf.variable_scope('layer_' + str(i)):
                        layer = hp.fully_connected(layer, layer_size)

            with tf.variable_scope('roll'):
                roll = layer
                for i, layer_size in enumerate(roll_sizes):
                    with tf.variable_scope('layer_' + str(i)):
                        roll = hp.fully_connected(roll, layer_size)
                with tf.variable_scope('output'):
                    roll = hp.fully_connected(roll, 1, bias=False, relu=False)

            with tf.variable_scope('yaw_pitch'):
                yaw_pitch = layer
                for i, layer_size in enumerate(yaw_pitch_sizes):
                    with tf.variable_scope('layer_' + str(i)):
                        yaw_pitch = hp.fully_connected(yaw_pitch, layer_size)

            with tf.variable_scope('yaw'):
                yaw = yaw_pitch
                for i, layer_size in enumerate(yaw_sizes):
                    with tf.variable_scope('layer_' + str(i)):
                        yaw = hp.fully_connected(yaw, layer_size)
                with tf.variable_scope('output'):
                    yaw = hp.fully_connected(yaw, 1, bias=False, relu=False)

            with tf.variable_scope('pitch'):
                pitch = yaw_pitch
                for i, layer_size in enumerate(pitch_sizes):
                    with tf.variable_scope('layer_' + str(i)):
                        pitch = hp.fully_connected(pitch, layer_size)
                with tf.variable_scope('output'):
                    pitch = hp.fully_connected(pitch, 1, bias=False, relu=False)

            with tf.variable_scope('output_layer'):
                attitude = tf.concat([yaw, pitch, roll], 1)
                attitude = tf.nn.dropout(attitude, keep_prob=self.keep_prob_placeholder)

        return attitude

    def build_model_two_channel(self):
        filter_sizes = [[13, 13], [5, 5]]
        channel_sizes = [10, 20]
        fully_connected_sizes = [512, 512]
        with tf.variable_scope('model'):
            with tf.variable_scope('convolution'):
                model = tf.concat([self.images_ref, self.images_new], axis=3)
                with tf.variable_scope('layer_1'):
                    model = hp.convolve(model, filter_sizes[0], self.image_shape[2]*2, channel_sizes[0])
                    model = tf.nn.relu(model)
                    model = hp.max_pool(model, [2, 2])
                with tf.variable_scope('layer_2'):
                    model = hp.convolve(model, filter_sizes[1], channel_sizes[0], channel_sizes[1])
                    model = tf.nn.relu(model)
                    model = hp.max_pool(model, [2, 2])
            with tf.variable_scope('fully_connected'):
                input_size = int(model.shape[1]*model.shape[2]*model.shape[3])
                model = tf.reshape(model, [-1, input_size])
                with tf.variable_scope('layer_1'):
                    weights = hp.weight_variables([input_size, fully_connected_sizes[0]])
                    biases = hp.bias_variables([fully_connected_sizes[0]])
                    model = tf.add(tf.matmul(model, weights), biases)
                    model = tf.nn.relu(model)
                with tf.variable_scope('layer_2'):
                    weights = hp.weight_variables([fully_connected_sizes[0], fully_connected_sizes[1]])
                    biases = hp.bias_variables([fully_connected_sizes[1]])
                    model = tf.add(tf.matmul(model, weights), biases)
                    model = tf.nn.relu(model)
            with tf.variable_scope('output_layer'):
                weights = hp.weight_variables([fully_connected_sizes[-1]] + self.label_shape)
                model = tf.matmul(model, weights)
                model = tf.nn.dropout(model, keep_prob=self.keep_prob_placeholder)
        return model

    def build_model_siamese(self):
        filter_sizes = [[13, 13], [5, 5]]
        channel_sizes = [8, 15]
        fully_connected_sizes = [512, 512]
        with tf.variable_scope('model'):
            with tf.variable_scope('branched_convolution'):
                with tf.variable_scope('layer_1') as scope:
                    ref = hp.convolve(self.images_ref, filter_sizes[0], self.image_shape[2], channel_sizes[0])
                    ref = tf.nn.relu(ref)
                    ref = hp.max_pool(ref, [2, 2])
                    scope.reuse_variables()
                    new = hp.convolve(self.images_new, filter_sizes[0], self.image_shape[2], channel_sizes[0])
                    new = tf.nn.relu(new)
                    new = hp.max_pool(new, [2, 2])
                with tf.variable_scope('layer_2') as scope:
                    ref = hp.convolve(ref, filter_sizes[1], channel_sizes[0], channel_sizes[1])
                    ref = tf.nn.relu(ref)
                    ref = hp.max_pool(ref, [2, 2])
                    scope.reuse_variables()
                    new = hp.convolve(new, filter_sizes[1], channel_sizes[0], channel_sizes[1])
                    new = tf.nn.relu(new)
                    new = hp.max_pool(new, [2, 2])
            with tf.variable_scope('fully_connected'):
                model = tf.concat([ref, new], axis=3)
                input_size = int(model.shape[1]*model.shape[2]*model.shape[3])
                model = tf.reshape(model, [-1, input_size])
                with tf.variable_scope('layer_1'):
                    weights = hp.weight_variables([input_size, fully_connected_sizes[0]])
                    biases = hp.bias_variables([fully_connected_sizes[0]])
                    model = tf.add(tf.matmul(model, weights), biases)
                    model = tf.nn.relu(model)
                with tf.variable_scope('layer_2'):
                    weights = hp.weight_variables([fully_connected_sizes[0], fully_connected_sizes[1]])
                    biases = hp.bias_variables([fully_connected_sizes[1]])
                    model = tf.add(tf.matmul(model, weights), biases)
                    model = tf.nn.relu(model)
            with tf.variable_scope('output_layer'):
                weights = hp.weight_variables([fully_connected_sizes[-1]] + self.label_shape)
                model = tf.matmul(model, weights)
                model = tf.nn.dropout(model, keep_prob=self.keep_prob_placeholder)
        return model

    def build_model_pseudo_siamese(self):
        filter_sizes = [[13, 13], [5, 5]]
        channel_sizes = [8, 15]
        fully_connected_sizes = [512, 512]
        with tf.variable_scope('model'):
            with tf.variable_scope('branched_convolution'):
                with tf.variable_scope('layer_1') as scope:
                    with tf.variable_scope('reference'):
                        ref = hp.convolve(self.images_ref, filter_sizes[0], self.image_shape[2], channel_sizes[0])
                        ref = tf.nn.relu(ref)
                        ref = hp.max_pool(ref, [2, 2])
                    with tf.variable_scope('new'):
                        new = hp.convolve(self.images_new, filter_sizes[0], self.image_shape[2], channel_sizes[0])
                        new = tf.nn.relu(new)
                        new = hp.max_pool(new, [2, 2])
                with tf.variable_scope('layer_2') as scope:
                    with tf.variable_scope('reference'):
                        ref = hp.convolve(ref, filter_sizes[1], channel_sizes[0], channel_sizes[1])
                        ref = tf.nn.relu(ref)
                        ref = hp.max_pool(ref, [2, 2])
                    with tf.variable_scope('new'):
                        new = hp.convolve(new, filter_sizes[1], channel_sizes[0], channel_sizes[1])
                        new = tf.nn.relu(new)
                        new = hp.max_pool(new, [2, 2])
            with tf.variable_scope('fully_connected'):
                model = tf.concat([ref, new], axis=3)
                input_size = int(model.shape[1]*model.shape[2]*model.shape[3])
                model = tf.reshape(model, [-1, input_size])
                with tf.variable_scope('layer_1'):
                    weights = hp.weight_variables([input_size, fully_connected_sizes[0]])
                    biases = hp.bias_variables([fully_connected_sizes[0]])
                    model = tf.add(tf.matmul(model, weights), biases)
                    model = tf.nn.relu(model)
                with tf.variable_scope('layer_2'):
                    weights = hp.weight_variables([fully_connected_sizes[0], fully_connected_sizes[1]])
                    biases = hp.bias_variables([fully_connected_sizes[1]])
                    model = tf.add(tf.matmul(model, weights), biases)
                    model = tf.nn.relu(model)
            with tf.variable_scope('output_layer'):
                weights = hp.weight_variables([fully_connected_sizes[-1]] + self.label_shape)
                model = tf.matmul(model, weights)
                model = tf.nn.dropout(model, keep_prob=self.keep_prob_placeholder)
        return model

    def build_model_siamese_regression(self):
        hidden_sizes = [3000, 1000]
        merged_sizes = [1000]
        with tf.variable_scope('model'):
            with tf.variable_scope('layer_1'):
                weights = hp.weight_variables([reduce(operator.mul, self.image_shape), hidden_sizes[0]])
                biases = hp.bias_variables([hidden_sizes[0]])
                ref = tf.reshape(self.images_ref, [-1, reduce(operator.mul, self.image_shape)])
                ref = tf.matmul(ref, weights) + biases
                ref = tf.nn.relu(ref)
                new = tf.reshape(self.images_new, [-1, reduce(operator.mul, self.image_shape)])
                new = tf.matmul(new, weights) + biases
                new = tf.nn.relu(new)
            with tf.variable_scope('layer_2'):
                weights = hp.weight_variables([hidden_sizes[0], hidden_sizes[1]])
                biases = hp.bias_variables([hidden_sizes[1]])
                ref = tf.matmul(ref, weights) + biases
                ref = tf.nn.relu(ref)
                new = tf.matmul(new, weights) + biases
                new = tf.nn.relu(new)
            with tf.variable_scope('layer_3'):
                model = tf.concat([ref, new], axis=1)
                weights = hp.weight_variables([hidden_sizes[1] * 2, merged_sizes[0]])
                biases = hp.bias_variables([merged_sizes[0]])
                model = tf.matmul(model, weights) + biases
                model = tf.nn.relu(model)
            with tf.variable_scope('output_layer'):
                weights = hp.weight_variables([merged_sizes[0], 3])
                model = tf.matmul(model, weights)
                model = tf.nn.dropout(model, keep_prob=self.keep_prob_placeholder)
        return model

    def train(self, train_path, validation_path=None, test_path=None):
        with tf.variable_scope('training'):
            sqr_dif = tf.reduce_sum(tf.square(self.model - self.labels), 1)
            mse = tf.reduce_mean(sqr_dif, name='mean_squared_error')
            angle_error = tf.reduce_mean(tf.sqrt(sqr_dif), name='mean_angle_error')
            tf.summary.scalar('angle_error', angle_error)
            optimizer = tf.train.AdamOptimizer().minimize(mse)

        summaries = tf.summary.merge_all()
        # if os.path.exists(self.conf.train_log_path):
        #     shutil.rmtree(self.conf.train_log_path)
        # os.mkdir(self.conf.train_log_path)

        print('Starting training\n')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # train_writer = tf.summary.FileWriter(self.conf.train_log_path, sess.graph)

            start_time = time.time()
            step = 0
            for epoch in range(1, self.conf.epochs + 1):
                epoch_angle_error = 0
                n_samples = self.initialize_iterator_with_set(sess, train_path, 'train')
                n_batches = int(n_samples / self.conf.batch_size)
                n_steps = n_batches * self.conf.epochs

                for batch in range(n_batches):
                    if step % max(int(n_steps / 1000), 1) == 0:
                        _, a, s = sess.run([optimizer, angle_error, summaries],
                                           feed_dict={self.keep_prob_placeholder: self.conf.keep_prob})
                        # train_writer.add_summary(s, step)
                        hp.log_step(step, n_steps, start_time, a)
                    else:
                        _, a = sess.run([optimizer, angle_error],
                                        feed_dict={self.keep_prob_placeholder: self.conf.keep_prob})

                    epoch_angle_error += a
                    step += 1

                hp.log_epoch(epoch, self.conf.epochs, epoch_angle_error / n_batches)
                if validation_path is not None:
                    self.error_for_set(sess, angle_error, validation_path, 'validation')

            # self.saver.save(sess, os.path.join(self.conf.train_log_path, 'model.ckpt'))
            if test_path is not None:
                self.error_for_set(sess, angle_error, test_path, 'test')
                # self.embeddings_for_set(sess, test_path)

    def predict(self, prediction_path):
        with tf.Session() as sess:
            try:
                self.saver.restore(sess, os.path.join(self.conf.train_log_path, 'model.ckpt'))
            except Exception as e:
                print(str(e))

            n_samples = self.initialize_iterator_with_set(sess, prediction_path, 'predict')
            predictions = np.ndarray([0] + self.label_shape)
            for _ in range(n_samples):
                prediction = sess.run(self.model, feed_dict={self.keep_prob_placeholder: 1.0})
                predictions = np.concatenate([predictions, prediction])

        return predictions

    def error_for_set(self, sess, error, path, name):
        n_samples = self.initialize_iterator_with_set(sess, path, 'predict')
        average_error = 0
        for _ in range(n_samples):
            average_error += sess.run(error, feed_dict={self.keep_prob_placeholder: 1.0}) / n_samples
        hp.log_generic(average_error, name)
        return average_error

    def embeddings_for_set(self, sess, path):
        n_samples = self.initialize_iterator_with_set(sess, path, 'predict')
        predictions = np.ndarray([0, 3])
        prediction_labels = np.ndarray([0, 3])
        for _ in range(n_samples):
            prediction, label = sess.run([self.model, self.labels], feed_dict={self.keep_prob_placeholder: 1.0})
            predictions = np.concatenate([predictions, prediction])
            prediction_labels = np.concatenate([prediction_labels, label])

        with tf.variable_scope('embedding'):
            embedding_var = tf.get_variable('embedding_var', shape=[predictions.shape[0], predictions.shape[1]],
                                            initializer=tf.constant_initializer(predictions))
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        metadata_file_path = os.path.join(self.conf.train_log_path, 'metadata.tsv')
        embedding.metadata_path = metadata_file_path

        # data labels
        with open(metadata_file_path, 'w') as f:
            for label in prediction_labels:
                f.write(str(label) + '\n')

        writer = tf.summary.FileWriter(self.conf.train_log_path)
        projector.visualize_embeddings(writer, config)
        embed_saver = tf.train.Saver([embedding_var])
        embed_saver.save(sess, os.path.join(self.conf.train_log_path, 'embeddding.ckpt'))

    def initialize_iterator_with_set(self, sess, path, set_type):
        images_ref, images_new, labels = hp.data_at_path(path)
        init = self.iterator.make_initializer(self.datasets[set_type])
        sess.run(init,
                 feed_dict={self.dataset_placeholders['images_ref']: images_ref,
                            self.dataset_placeholders['images_new']: images_new,
                            self.dataset_placeholders['labels']: labels})
        return len(images_ref)
