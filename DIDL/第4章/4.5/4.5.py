import torch
from torch import nn
from d2l import torch as d2l


n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    """初始化模型参数"""
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    """计算L2范数"""
    return torch.sum(w.pow(2)) / 2


def train(lambd):
    w ,b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003

    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])

    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加L2范数惩罚项
            l = loss(net(X), y) + lambd * l2_penalty(w)
            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))


def train_concise(wd):
    """简洁实现"""
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    num_epochs = 100
    for param in net.parameters():
        param.data.normal_(0, 1)

    
    loss = nn.MSELoss()
    # 只对权重衰减，不对偏置衰减
    trainer = torch.optim.SGD([
        {'params': net[0].weight, 'weight_decay': wd},
        {'params': net[0].bias}], lr=0.003)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))

def train_L1(l1_lambda):
    """简洁实现"""
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    num_epochs = 100
    for param in net.parameters():
        param.data.normal_(0, 1)

    
    loss = nn.MSELoss()
    # 只对权重衰减，不对偏置衰减
    trainer = torch.optim.SGD([
        {'params': net[0].weight, 'weight_decay': 0},
        {'params': net[0].bias}], lr=0.003)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            prediction_loss = loss(net(X), y)
            l = prediction_loss + l1_lambda * torch.sum(torch.abs(net[0].weight)) 
            # 后面的部分为L1正则化项
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))


if __name__ == '__main__':
    # train_L1(l1_lambda=0)  # 不使用权重衰减
    # d2l.plt.show()
    train_L1(l1_lambda=3)  # 使用权重衰减
    d2l.plt.show()
