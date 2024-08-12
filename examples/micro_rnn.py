from axon.micro import value, nn

xs = [
  [2.0, 3.0, -1.0],
  [3.0, 0.0, -0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]
model = nn.RNN(input_size=3, hidden_size=4, output_size=1)

epochs = 100
learning_rate = 0.001

for k in range(epochs):
  out = [model(x) for x in xs]
  loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, out))

  model.zero_grad()
  loss.backward()

  for p in model.parameters():
    p.data -= learning_rate * p.grad

  print(k, " -> ", loss.data)


print([model(x) for x in xs])