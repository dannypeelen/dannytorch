import dannytorch as dt
import dannytorch.nn as nn
import dannytorch.optim as optim
import dannytorch.optim.scheduler as scheduler
from datasets import load_dataset

#================LLM TEST=====================

#use cosmopedia-100k?

#load dataset
ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
print(ds)
#things to remember:
# - good balance of num_epochs and epoch size
# - keep "prompt", "text" sections

# #=============MNIST TEST====================
# import kagglehub
# from keras.datasets import mnist

# # Download latest version
# # path = kagglehub.dataset_download("hojjatk/mnist-dataset")
# # print(f"Path: {path}")

# mlp = nn.MLP(784, [64, 64, 10])
# cse = nn.CrossEntropyLoss()

# adam = optim.Adam(mlp.parameters())
# scheduler = scheduler.ExponentialLR(adam, step_size=1)

# (train_X, train_y), (test_X, test_y) = mnist.load_data()

# batch_size = 32

# for epoch in range(1, 5):
#     for start in range(0, len(train_X), batch_size):
#         end = min(start + batch_size, len(train_X))

#         preds = [mlp(x.flatten() / 255.0) for x in train_X[start:end]]
#         batch_labels = train_y[start:end]

#         loss = cse(preds, batch_labels)

#         mlp.zero_grad()
#         loss.backward()


#         # for param in mlp.parameters():
#         #     param.data = param.data - 0.01 * param.grad

#         adam.step()

#         if start % 6400 == 0:
#             print(f"Epoch {epoch}, step {start // batch_size}: Loss={loss.data:.4f}")

#     scheduler.step()
#FROM 7.8 to 0.12

#=============MLP TEST======================

# in_vals = [
#     [1, 0, 1, 1],
#     [0, 0.5, 0.5, 1],
#     [0, 0, 1, 1],
#     [1, 0, 0, 0.5]
# ]

# out_vals = [1, 0, 0, 0.5]

# mlp = nn.MLP(4, [4,4,1])


# for i in range(1,21):
#     pred = [mlp(x) for x in in_vals]
#     print(f"Preds:{pred}")
#     loss = sum((ypred-yout) ** 2 for yout, ypred in zip(out_vals, pred))
#     print(f"Loss: {loss}")

#     #backprop
#     mlp.zero_grad()
#     loss.backward()
#     for p in mlp.parameters():
#         p.data = p.data + -0.01 * p.grad

#     input(f"Press Enter to step to {i+1} ")
# loss goes from 51 to .63!!!!!!

#============BACKWARD TEST=====================

# t1 = dt.tensor(0.5)
# t2 = dt.tensor(0.6)
# t3 = dt.tensor(0.7)

# t4 = t1*t2+t3
# print(t4)
# t4.backward()
# t4.backward()
# print(f"{t1} {t2} {t3} {t4}")




    


