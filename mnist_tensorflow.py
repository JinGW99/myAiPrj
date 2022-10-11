import tensorflow as tf

#1 mnist 데이터셋 가져오기
mnist = tf.keras.datasets.mnist
#mnist로부터 학습용데이터와 검증용 테스트 데이터 가져오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#2데이터 전처리
#0~255, 0사이의 값을 갖는 픽셀값들을 0~1.0 사이의 값을갖도록 변환
x_train, x_test = x_train/255.0, x_test/255.0

#3.학습 모델 구성 방법 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(512, activation=tf.nn.relu),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("정확도:", test_acc)

model.save("model/myModel.h5")