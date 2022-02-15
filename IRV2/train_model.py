from IRV2 import data_utils
from IRV2 import irv2
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

if __name__ == '__main__':

    datagen = data_utils.get_preprocessing_datagen()
    train_path, test_path = '', ''
    image_size = 299
    batch_size = 50
    save_path = ''
    train_size, test_size = 0, 0

    print("\nTrain Batches: ")
    train_batches = datagen.flow_from_directory(directory=train_path,
                                                target_size=(image_size, image_size),
                                                batch_size=batch_size,
                                                shuffle=True)

    print("\nTest Batches: ")
    test_batches = datagen.flow_from_directory(test_path,
                                               target_size=(image_size, image_size),
                                               batch_size=batch_size,
                                               shuffle=False)

    model = irv2.get_model(7)
    class_weights = {
        0: 1.0,  # akiec
        1: 1.0,  # bcc
        2: 1.0,  # bkl
        3: 1.0,  # df
        4: 5.0,  # mel
        5: 1.0,  # nv
        6: 1.0,  # vasc
    }

    checkpoint = ModelCheckpoint(filepath=save_path + 'saved_model.hdf5', monitor='val_accuracy', save_best_only=True,
                                 save_weights_only=True)

    Earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=30, min_delta=0.001)
    history = model.fit(train_batches,
                        steps_per_epoch=(train_size / 10),
                        epochs=150,
                        verbose=2,
                        validation_data=test_batches, validation_steps=test_size / batch_size,
                        callbacks=[checkpoint, Earlystop], class_weight=class_weights)

    
