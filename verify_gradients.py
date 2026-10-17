"""Check the manual backpropagation formula against autograd on a tiny batch."""
import torch

torch.manual_seed(42)
x = torch.randn(7, 12, dtype=torch.float64)
y = torch.tensor([0, 1, 2, 0, 1, 2, 0])
w1 = torch.randn(12, 5, dtype=torch.float64, requires_grad=True)
b1 = torch.randn(5, dtype=torch.float64, requires_grad=True)
w2 = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
b2 = torch.randn(3, dtype=torch.float64, requires_grad=True)
h = torch.sigmoid(x @ w1 + b1)
logits = h @ w2 + b2
loss = torch.nn.functional.cross_entropy(logits, y)
with torch.no_grad():
    dz2 = (logits.softmax(1) - torch.nn.functional.one_hot(y, 3)) / len(y)
    dz1 = (dz2 @ w2.T) * h * (1 - h)
    manual = [x.T @ dz1, dz1.sum(0), h.T @ dz2, dz2.sum(0)]
loss.backward()
for parameter, gradient in zip([w1, b1, w2, b2], manual):
    torch.testing.assert_close(parameter.grad, gradient, rtol=1e-10, atol=1e-12)
print("All four manual gradients match autograd in float64.")
