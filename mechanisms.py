import math
import random
import torch
import torch.nn.functional as F


def calculate_b(y, d):
    """
    计算b的值
    """
    return torch.floor((y * torch.exp(y) - torch.exp(y) + 1) / (2 * torch.exp(y) * (torch.exp(y) - 1 - y))) * d


def calculate_p_and_q(y, b, d):
    """
    计算p和q的值
    """
    denominator = (2 * b + 1) * torch.exp(y) + d - 1
    p = torch.exp(y) / denominator
    q = 1 / denominator
    return p, q


def SW_mechanism(v, y, D):
    """
    SW机制函数
    """
    d = len(D)
    b = calculate_b(y, d)
    p, q = calculate_p_and_q(y, b, d)

    D_star = torch.arange(1, d + 2 * b + 1)

    diff = torch.abs(v - D_star)
    prob = torch.where(diff <= b, p, q)

    rand_prob = torch.rand(prob.size())
    mask = rand_prob < prob

    return torch.where(mask, D_star, v).max().item()


class Mechanism:
    def __init__(self, eps, input_range, **kwargs):
        self.eps = eps
        self.alpha, self.beta = input_range

    def __call__(self, x):
        raise NotImplementedError


class Laplace(Mechanism):
    def __call__(self, x):
        d = x.size(1)
        sensitivity = (self.beta - self.alpha) * d
        scale = torch.ones_like(x) * (sensitivity / self.eps)
        out = torch.distributions.Laplace(x, scale).sample()
        # out = torch.clip(out, min=self.alpha, max=self.beta)
        return out


class MultiBit(Mechanism):
    def __init__(self, *args, m='best', **kwargs):
        super().__init__(*args, **kwargs)
        self.m = m

    def __call__(self, x):
        n, d = x.size()
        if self.m == 'best':
            m = int(max(1, min(d, math.floor(self.eps / 2.18))))
        elif self.m == 'max':
            m = d
        else:
            m = self.m

        # sample features for perturbation
        BigS = torch.rand_like(x).topk(m, dim=1).indices
        s = torch.zeros_like(x, dtype=torch.bool).scatter(1, BigS, True)
        del BigS

        # perturb sampled features
        em = math.exp(self.eps / m)
        p = (x - self.alpha) / (self.beta - self.alpha)
        p = (p * (em - 1) + 1) / (em + 1)
        t = torch.bernoulli(p)
        x_star = s * (2 * t - 1)
        del p, t, s

        # unbiase the result
        x_prime = d * (self.beta - self.alpha) / (2 * m)
        x_prime = x_prime * (em + 1) * x_star / (em - 1)
        x_prime = x_prime + (self.alpha + self.beta) / 2
        user_levels_perturb = None
        return x_prime, user_levels_perturb


def generate_data_list(n, D):
    """
    生成一个包含 n 个随机值的列表，每个值都从D域中随机选择。
    """
    return [random.choice(D) for _ in range(n)]


def generate_eps_list(data_list, eqs):
    """
    根据data_list生成eps_list。
    """
    eps_mapping = {1: eqs, 2: eqs * 2, 3: eqs * 4, 4: eqs * 8, 5: eqs * 16, 6: eqs * 32, 7: eqs * 64, 8: eqs * 128,
                   9: eqs * 256}  # 映射关系

    eps_list = [eps_mapping[val] for val in data_list]

    return eps_list


class PersonalizedPerturbation(Mechanism):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_device(self, tensor, device):
        return tensor.to(device)

    def __call__(self, x):
        n, d = x.size()

        device = x.device

        # Generate random privacy levels (1 or 2) for each user based on level1_frac
        # level1_count = int(n * self.lta)
        # level2_count = n - level1_count
        eps_mapping1 = {1: self.eps, 2: self.eps * 2, 3: self.eps * 4, 4: self.eps * 8, 5: self.eps * 16}
        D = [1, 2, 3, 4, 5]
        user_levels = generate_data_list(n, D)
        user_epsilons = generate_eps_list(user_levels, self.eps)
        user_levels = torch.Tensor(user_levels)
        user_epsilons = torch.Tensor(user_epsilons)
        delta = 0.5
        feature_epsilons = (1 - delta) * user_epsilons
        level_epsilons = 0.5 * delta * user_epsilons
        D = set(D)
        user_levels_perturb = []
        for user_level, y in zip(user_levels, level_epsilons):
            result = SW_mechanism(user_level, y, D)
            user_levels_perturb.append(result)
        feature_epsilons_perturb = (1 - delta) * torch.Tensor(generate_eps_list(user_levels_perturb, self.eps))

        m_values = self.calculate_m_values(feature_epsilons, d)
        m_values_perturb = []
        HH = max(1, int(eps_mapping1[5] / 2.18))
        H = set(range(1, HH + 1))
        for m_value, y in zip(m_values, level_epsilons):
            m_value = torch.tensor(m_value, dtype=torch.float32)
            result = SW_mechanism(m_value, y, H)
            m_values_perturb.append(result)
        m_values = torch.tensor(m_values_perturb, dtype=torch.int)
        BigS = torch.stack([torch.randperm(d, device=device)[:max(m_values)] for _ in range(n)])

        s = torch.zeros_like(x, dtype=torch.bool, device=device).scatter(1, BigS, True)

        em_values = torch.exp(feature_epsilons / m_values.float())
        em_values_perturb = torch.exp(feature_epsilons_perturb / m_values.float())
        p = (x - self.alpha) / (self.beta - self.alpha)
        p = (p * (em_values.view(-1, 1) - 1) + 1) / (em_values.view(-1, 1) + 1)
        t = torch.bernoulli(p)
        x_star = s * (2 * t - 1)

        x_prime = d * (self.beta - self.alpha) / (2 * m_values.view(-1, 1))
        x_prime = x_prime * (em_values_perturb.view(-1, 1) + 1) * x_star / (em_values_perturb.view(-1, 1) - 1)
        x_prime = x_prime + (self.alpha + self.beta) / 2

        user_levels_perturb = torch.tensor(user_levels_perturb, dtype=torch.int32)
        return x_prime, user_levels_perturb

    def calculate_m_values(self, epsilons, d):
        m_values = torch.clamp_min(torch.floor(epsilons / 2.18).int(), 1)
        return m_values


class OneBit(MultiBit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, m='max', **kwargs)


def RandomizedResopnse(eps, d, data):
    q = 1.0 / (torch.exp(eps) + d - 1)
    p = q * torch.exp(eps)
    pr = data * p.unsqueeze(1) + (1 - data) * q.unsqueeze(1)
    out = torch.multinomial(pr, num_samples=1)
    return F.one_hot(out.squeeze(), num_classes=d)


supported_feature_mechanisms = {
    'mbm': MultiBit,
    '1bm': OneBit,
    'lpm': Laplace,
    'ppm': PersonalizedPerturbation
}
