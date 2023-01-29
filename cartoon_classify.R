require(keras)
img_height = 32
img_width = 32
batch_size = 8

train_dir = "~/cartoons/train"
test_dir = "~/cartoons/test"

train_generator = flow_images_from_directory(train_dir,
                                             generator = image_data_generator(),
                                             target_size = c(img_width,img_height),
                                             color_mode = "rgb",
                                             class_mode = "categorical",
                                             batch_size = batch_size,
                                             shuffle = T,seed=123)

validation_generator = flow_images_from_directory(test_dir,
                                             generator = image_data_generator(),
                                             target_size = c(img_width,img_height),
                                             color_mode = "rgb",
                                             classes = NULL,
                                             class_mode = "categorical",
                                             batch_size = batch_size,
                                             shuffle = F,seed=123)
train_samples = 616
validation_samples = 156

base_model = application_vgg16(weights = "imagenet", include_top = F)

predictions = base_model$output %>%
  layer_global_average_pooling_2d(trainable=F) %>%
  layer_dense(64, trainable = F) %>%
  layer_activation("relu", trainable=F) %>%
  layer_dropout(0.4,trainable=F) %>%
  layer_dense(2, trainable = F) %>%
  layer_activation("softmax",trainable = T)

model = keras_model(inputs = base_model$input, outputs = predictions)

model %>% compile(loss="categorical_crossentropy",
                  optimizer=optimizer_rmsprop(learning_rate = 0.001,decay=1e-6),
                  metrics="accuracy")

hist = model %>% fit(train_generator,
                    steps_per_epoch = as.integer(train_samples/batch_size),
                   epochs = 50,
                   validation_data = validation_generator,
                    validation_steps = as.integer(validation_samples/batch_size),
                     verbose=2)
histDF = data.frame(acc=unlist(hist$metrics$accuracy),val_acc=unlist(hist$metrics$val_accuracy),
                    loss=unlist(hist$metrics$loss),val_loss=unlist(hist$metrics$val_loss))
