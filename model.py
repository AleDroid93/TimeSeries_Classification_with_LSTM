import tensorflow as tf

class MyLSTM(tf.keras.Model):
    def __init__(self, units, n_classes, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='lstmNetwork',
                 **kwargs):
        # chiamata al costruttore della classe padre, Model
        super(MyLSTM, self).__init__(name=name, **kwargs)
        # definizione dei layers del modello
        self.lstm1 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units), return_sequences=True, return_state=True,
                                         name="lstm1")
        self.lstm2 = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units), name="lstm2")
        self.model_output = tf.keras.layers.Dense(units=n_classes, activation=output_activation)

    def call(self, inputs, training=False):
        # definisco il flusso, che la rete rappresentata dal modello, deve seguire.
        inputs = self.lstm1(inputs, training=training)
        inputs = self.lstm2(inputs, training=training)
        return self.model_output(inputs)