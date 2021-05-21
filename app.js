const tf = require("@tensorflow/tfjs")

const temp = tf.tensor([20, 21, 22, 23, 30]);
const sold = tf.tensor([40, 42, 44, 46, 60]);

const X = tf.input({ shape: [1] });
const Y = tf.layers.dense({ units: 1 }).apply(X);
const model = tf.model({ inputs: X, outputs: Y });
const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError }
model.compile(compileParam);

const fitParam = {
    epochs: 10000, 
    callbacks:{
        onEpochEnd: function(epoch, loss) {
            console.log(`${epoch}번 학습중, 평균 오차 ${Math.sqrt(loss.loss)}`)
        }
    } 
}
model.fit(temp, sold, fitParam).then(function (result) {
    model.predict(tf.tensor([15,16,17, 18, 19])).print();
});