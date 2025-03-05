import torch

x = torch.ones(5) #input tensor
y = torch.ones(3) #expected tensor
w = torch.randn(5,3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x,w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)

#변화도 추적 멈추기 모델 학습 이후 데이터를 단순 적용할때에 순전파 연산만 필요하는 경우가 있다.
z = torch.matmul(x,w) + b
print(z.requires_grad)

with torch.no_grad(): #gradient 비활성화
    z = torch.matmul(x,w) + b
print(z.requires_grad)


z = torch.matmul(x,w) + b
z_det = z.detach()
print(z_det.requires_grad)

#야코비안 곱
inp =torch.eye(4,5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True) #backward는 변화도를 누적해서 알려줌
print(f"Second call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients]n{inp.grad}")