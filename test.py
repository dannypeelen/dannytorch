import dannytorch as dt
import dannytorch.llm as llm
import dannytorch.optim as optim
import dannytorch.optim.scheduler as scheduler
import dannytorch.nn as nn
import numpy as np

#================LLM TEST=====================
CORPUS = """
It was the best of times, it was the worst of times, it was the age of wisdom,
it was the age of foolishness, it was the epoch of belief, it was the epoch of
incredulity, it was the season of Light, it was the season of Darkness, it was
the spring of hope, it was the winter of despair, we had everything before us,
we had nothing before us, we were all going direct to Heaven, we were all going
direct the other way. To be, or not to be, that is the question: whether 'tis
nobler in the mind to suffer the slings and arrows of outrageous fortune, or to
take arms against a sea of troubles and by opposing end them. It is a truth
universally acknowledged that a single man in possession of a good fortune must
be in want of a wife. Call me Ishmael. Some years ago, never mind how long
precisely, having little money in my pocket and nothing particular to interest
me on shore, I thought I would sail about a little and see the watery part of
the world. In the beginning God created the heavens and the earth, and the earth
was without form and void, and darkness was upon the face of the deep.
"""

CONTEXT = 16
EPOCHS  = 500

chars = sorted(set(CORPUS))
ctoi  = {c: i for i, c in enumerate(chars)}
itoc  = {i: c for c, i in ctoi.items()}
V     = len(chars)
data  = [ctoi[c] for c in CORPUS]
pairs = [(data[i:i+CONTEXT], data[i+CONTEXT]) for i in range(len(data)-CONTEXT)]

model   = llm.Transformer(vocab_size=V, d_model=64, n_heads=4, n_blocks=2, max_seq_len=CONTEXT)
opt     = optim.Adam(model.parameters(), lr=1e-3)
sched   = scheduler.CosineAnnealingLR(opt, EPOCHS)
loss_fn = nn.CrossEntropyLoss()

def last_logit(out):
    t = dt.tensor(out.data[0, -1, :].copy(), (out,))
    def _bwd(): out.grad[0, -1, :] += t.grad
    t._backward = _bwd
    return t

def sample(seed="It was", n=80):
    ctx = ([0]*(CONTEXT-len(seed)) + [ctoi.get(c,0) for c in seed])[-CONTEXT:]
    out = list(seed)
    for _ in range(n):
        lg = last_logit(model(dt.tensor(np.array([ctx]), requires_grad=False)))
        p  = np.exp(lg.data - lg.data.max()); p /= p.sum()
        nxt = np.random.choice(V, p=p)
        out.append(itoc[nxt]); ctx = ctx[1:] + [nxt]
    return "".join(out)

print(f"vocab={V}  pairs={len(pairs)}  ctx={CONTEXT}  epochs={EPOCHS}\n")
print(f"[ep   0] init | {sample()!r}\n")

for ep in range(1, EPOCHS+1):
    np.random.shuffle(pairs); epoch_loss = 0.0
    for xs, y in pairs:
        pred = last_logit(model(dt.tensor(np.array([xs]), requires_grad=False)))
        loss = loss_fn([pred], [y])
        model.zero_grad(); loss.backward(); sched.step()
        epoch_loss += float(loss.data)
    if ep % 50 == 0:
        print(f"[ep {ep:3d}] loss {epoch_loss/len(pairs):.4f} | {sample()!r}")


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




    


