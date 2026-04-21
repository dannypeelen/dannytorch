import dannytorch as dt
import dannytorch.llm as llm
import dannytorch.optim as optim
import dannytorch.optim.scheduler as scheduler
import dannytorch.nn as nn
import numpy as np
import gc

#=============BIGGER LLM TEST=================

CORPUS2 = """
It was the best of times, it was the worst of times, it was the age of wisdom,
it was the age of foolishness, it was the epoch of belief, it was the epoch of
incredulity, it was the season of Light, it was the season of Darkness, it was
the spring of hope, it was the winter of despair, we had everything before us,
we had nothing before us, we were all going direct to Heaven, we were all going
direct the other way. There were a king with a large jaw and a queen with a plain
face, on the throne of England; there were a king with a large jaw and a queen
with a fair face, on the throne of France. In both countries it was clearer than
crystal to the lords of the State preserves of loaves and fishes, that things in
general were settled for ever. It was the year of Our Lord one thousand seven
hundred and seventy-five. Spiritual revelations were conceded to England at that
favoured period, as at this. To be, or not to be, that is the question: whether
tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or
to take arms against a sea of troubles and by opposing end them. To die, to sleep,
no more; and by a sleep to say we end the heartache and the thousand natural shocks
that flesh is heir to. All the world is a stage, and all the men and women merely
players. They have their exits and their entrances, and one man in his time plays
many parts. It is a truth universally acknowledged that a single man in possession
of a good fortune must be in want of a wife. However little known the feelings or
views of such a man may be on his first entering a neighbourhood, this truth is so
well fixed in the minds of the surrounding families that he is considered as the
rightful property of some one or other of their daughters. Call me Ishmael. Some
years ago, never mind how long precisely, having little money in my pocket and
nothing particular to interest me on shore, I thought I would sail about a little
and see the watery part of the world. It is a way I have of driving off the spleen
and regulating the circulation. Whenever I find myself growing grim about the mouth,
whenever it is a damp drizzly November in my soul, whenever I find myself involuntarily
pausing before coffin warehouses and bringing up the rear of every funeral I meet,
then I account it high time to get to sea as soon as I can. In the beginning God
created the heavens and the earth, and the earth was without form and void, and
darkness was upon the face of the deep. And the Spirit of God moved upon the face
of the waters. And God said, Let there be light: and there was light. And God saw
the light, that it was good: and God divided the light from the darkness.
"""

CONTEXT2 = 32
EPOCHS2   = 200

chars2 = sorted(set(CORPUS2))
ctoi2  = {c: i for i, c in enumerate(chars2)}
itoc2  = {i: c for c, i in ctoi2.items()}
V2     = len(chars2)
data2  = [ctoi2[c] for c in CORPUS2]
# (input_T, target_T) — targets are input shifted by 1, giving T loss signals per sequence
pairs2  = [(data2[i:i+CONTEXT2], data2[i+1:i+CONTEXT2+1]) for i in range(len(data2)-CONTEXT2)]
BATCH2  = 32

model2   = llm.Transformer(vocab_size=V2, d_model=128, n_heads=4, n_blocks=3, max_seq_len=CONTEXT2)
opt2     = optim.Adam(model2.parameters(), lr=5e-4)
sched2   = scheduler.CosineAnnealingLR(opt2, EPOCHS2, eta_min=1e-5)
loss_fn2 = nn.CrossEntropyLoss(batched=True)

def sample2(seed="It was ", n=120):
    model2.eval()
    ctx = ([0]*(CONTEXT2-len(seed)) + [ctoi2.get(c,0) for c in seed])[-CONTEXT2:]
    out = list(seed)
    for _ in range(n):
        logits = model2(dt.tensor(np.array([ctx]), requires_grad=False))
        lg = logits.data[0, -1, :]
        p  = np.exp(lg - lg.max()); p /= p.sum()
        nxt = np.random.choice(V2, p=p)
        out.append(itoc2[nxt]); ctx = ctx[1:] + [nxt]
    model2.train()
    gc.collect()
    return "".join(out)

n_batches2 = (len(pairs2) + BATCH2 - 1) // BATCH2
print(f"vocab={V2}  pairs={len(pairs2)}  ctx={CONTEXT2}  batch={BATCH2}  epochs={EPOCHS2}\n")
print(f"[ep   0] init | {sample2()!r}\n")

for ep in range(1, EPOCHS2+1):
    np.random.shuffle(pairs2); epoch_loss = 0.0
    for b in range(n_batches2):
        batch  = pairs2[b*BATCH2:(b+1)*BATCH2]
        xs     = np.array([p[0] for p in batch])           # (B, T)
        ys     = np.array([p[1] for p in batch]).flatten() # (B*T,)
        logits = model2(dt.tensor(xs, requires_grad=False)) # (B, T, V)
        B, T, Vd = logits.data.shape
        loss   = loss_fn2(logits.reshape(B*T, Vd), ys)
        model2.zero_grad(); loss.backward(); opt2.step()
        epoch_loss += float(loss.data)
        bar = ('█' * int((b+1) / n_batches2 * 20)).ljust(20)
        print(f"\r[ep {ep:3d}/{EPOCHS2}] |{bar}| {b+1}/{n_batches2}  loss={epoch_loss/(b+1):.4f}  lr={opt2.lr:.2e}", end='', flush=True)
    sched2.step()
    gc.collect()
    if ep % 25 == 0:
        print(f"\n[ep {ep:3d}] loss {epoch_loss/n_batches2:.4f} | {sample2()!r}")
    else:
        print()

#================LLM TEST=====================
# CORPUS = """
# It was the best of times, it was the worst of times, it was the age of wisdom,
# it was the age of foolishness, it was the epoch of belief, it was the epoch of
# incredulity, it was the season of Light, it was the season of Darkness, it was
# the spring of hope, it was the winter of despair, we had everything before us,
# we had nothing before us, we were all going direct to Heaven, we were all going
# direct the other way. To be, or not to be, that is the question: whether 'tis
# nobler in the mind to suffer the slings and arrows of outrageous fortune, or to
# take arms against a sea of troubles and by opposing end them. It is a truth
# universally acknowledged that a single man in possession of a good fortune must
# be in want of a wife. Call me Ishmael. Some years ago, never mind how long
# precisely, having little money in my pocket and nothing particular to interest
# me on shore, I thought I would sail about a little and see the watery part of
# the world. In the beginning God created the heavens and the earth, and the earth
# was without form and void, and darkness was upon the face of the deep.
# """

# CONTEXT = 16
# EPOCHS  = 500

# chars = sorted(set(CORPUS))
# ctoi  = {c: i for i, c in enumerate(chars)}
# itoc  = {i: c for c, i in ctoi.items()}
# V     = len(chars)
# data  = [ctoi[c] for c in CORPUS]
# pairs = [(data[i:i+CONTEXT], data[i+CONTEXT]) for i in range(len(data)-CONTEXT)]

# model   = llm.Transformer(vocab_size=V, d_model=64, n_heads=4, n_blocks=2, max_seq_len=CONTEXT)
# opt     = optim.Adam(model.parameters(), lr=1e-3)
# sched   = scheduler.CosineAnnealingLR(opt, EPOCHS)
# loss_fn = nn.CrossEntropyLoss()

# def last_logit(out):
#     t = dt.tensor(out.data[0, -1, :].copy(), (out,))
#     def _bwd(): out.grad[0, -1, :] += t.grad
#     t._backward = _bwd
#     return t

# def sample(seed="It was ", n=80):
#     model.eval()
#     ctx = ([0]*(CONTEXT-len(seed)) + [ctoi.get(c,0) for c in seed])[-CONTEXT:]
#     out = list(seed)
#     for _ in range(n):
#         lg = last_logit(model(dt.tensor(np.array([ctx]), requires_grad=False)))
#         p  = np.exp(lg.data - lg.data.max()); p /= p.sum()
#         nxt = np.random.choice(V, p=p)
#         out.append(itoc[nxt]); ctx = ctx[1:] + [nxt]
#     model.train()
#     gc.collect()
#     return "".join(out)

# print(f"vocab={V}  pairs={len(pairs)}  ctx={CONTEXT}  epochs={EPOCHS}\n")
# print(f"[ep   0] init | {sample()!r}\n")

# for ep in range(1, EPOCHS+1):
#     np.random.shuffle(pairs); epoch_loss = 0.0
#     for i, (xs, y) in enumerate(pairs):
#         pred = last_logit(model(dt.tensor(np.array([xs]), requires_grad=False)))
#         loss = loss_fn([pred], [y])
#         model.zero_grad(); loss.backward(); opt.step()
#         epoch_loss += float(loss.data)
#         done = i + 1
#         bar  = ('█' * int(done / len(pairs) * 20)).ljust(20)
#         print(f"\r[ep {ep:3d}/{EPOCHS}] |{bar}| {done}/{len(pairs)}  loss={epoch_loss/done:.4f}  lr={opt.lr:.2e}", end='', flush=True)
#     sched.step()
#     gc.collect()
#     if ep % 50 == 0:
#         print(f"\n[ep {ep:3d}] loss {epoch_loss/len(pairs):.4f} | {sample()!r}")
#     else:
#         print()


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




    


