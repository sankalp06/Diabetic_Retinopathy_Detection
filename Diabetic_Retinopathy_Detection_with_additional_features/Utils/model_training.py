from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

def create_additional_input(shape):
    return Input(shape=shape)

def create_combined_model(model_input, model_output, additional_input, num_classes=5, additional_layers=None):
    combined = concatenate([model_output, additional_input])
    if additional_layers is not None:
        for layer in additional_layers:
            combined = layer(combined)
    else:
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.5)(combined)
    final_output = Dense(num_classes, activation='softmax')(combined)
    return Model(inputs=[model_input, additional_input], outputs=final_output)

def compile_and_train_model(model, X_train_img, X_train_feat, y_train, X_test_img, X_test_feat, y_test, callbacks, optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'], epochs=12, batch_size=64):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(
        [X_train_img, X_train_feat], y_train,
        validation_data=([X_test_img, X_test_feat], y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    return history


def evaluate_and_save_model(model, X_test_img, X_test_feat, y_test, model_save_path, metrics_save_path):
    # Evaluate the model
    loss, accuracy = model.evaluate([X_test_img, X_test_feat], y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Predict classes
    y_pred = model.predict([X_test_img, X_test_feat]).argmax(axis=1)

    # Calculate precision, recall, and classification report
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test.argmax(axis=1), y_pred, average='weighted')
    cls_report = classification_report(y_test.argmax(axis=1), y_pred)

    # Save metrics to text file
    with open(metrics_save_path, 'w') as f:
        f.write(f'Loss: {loss}, Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}, Recall: {recall}\n')
        f.write('Classification Report:\n')
        f.write(cls_report)

    # Save the model
    model.save(model_save_path)
