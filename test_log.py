import torch

x = torch.tensor(-100)

# Wrong:
a = torch.log(torch.exp(x)*torch.exp(x))
print("problem:", a)

# Right:
a = torch.log(torch.exp(x)) + torch.log(torch.exp(x))
print("No problem:", a)

# Right style with torch function:
x = torch.tensor([-100, -100])
a = torch.logsumexp(x, dim=0)
print(a)
