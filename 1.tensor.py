import torch
import numpy as np

#텐서 초기화
#데이터로부터 직접 생성하기
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(f"{x_data.type()=}")
#numpy로부터 생성하기
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) #x_data의 속성을 유지
print(f"One tensor: \n{x_ones}\n")
#명시적으로 재정의하지 않는다면, 인자로 주어진 텐서의 속성을 유지한다.
x_rand = torch.rand_like(x_data, dtype=torch.float) #x_data의 속성을 덮어쓴다
print(f"Random Tensor: \n{x_rand}\n")
print(f"{x_rand.type()}")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n{rand_tensor}\n")
print(f"Ones Tensor: \n{ones_tensor}\n")
print(f"Zeros Tensor: \n{zeros_tensor}\n")

#텐서의 속성은 텐서의 모양, 자료형 및 어느 장치에 저장되는지를 나타낸다.
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#GPU가 존재하면 텐서를 이동한다.
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

#Numpy식의 표준 인덱싱과 슬라이싱
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[...,-1]}")
tensor[:,1] = 0
print(tensor)

#텐서 합치기
t1 = torch.cat([tensor,tensor,tensor], dim=1)
print(t1)

#산술 연산
#두 텐서간의 행렬 곱을 계산, y1,y2,y3는 모두 같은 값을 가진다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

#요소곱 z1,z2,z3는 모두 같은 값을 가진다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(z1)
torch.mul(tensor,tensor, out=z3)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#바꿔치기 연산
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n,1, out=n)
print(f"t: {t}")
print(f"n: {n}")