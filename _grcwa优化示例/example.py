import sys
import time
import nlopt
import pandas as pd
import autograd
import autograd.numpy as np
import grcwa
grcwa.set_backend('autograd')  # important!!

import my_draw
import matplotlib.pyplot as plt

LOG_ON = False # 是否在 log.txt 记录某些过程信息

# 假设这里出现的长度单位皆为 um

# 网格的划分数目
Nx, Ny = 100, 100

# RCWA 级数
nG = 51

# 光源参数
plane_wv = {'p_amp': 1 / np.sqrt(2), 's_amp': 1 / np.sqrt(2), 'p_phase': 0, 's_phase': 0}
# plane_wv = {'p_amp': 1, 's_amp': 0, 'p_phase': 0, 's_phase': 0}

# 材料参数
e_vacuum = 1 + 0j

nk_0 = pd.read_csv('materials\\Si_0.25_14.txt', sep = ' ').to_numpy()
nk_1 = pd.read_csv('materials\\SiO2_7_50.txt', sep = '\t').to_numpy()
nk_2 = pd.read_csv('materials\\Au_0.3_24.93.txt', sep = ' ').to_numpy()
print(nk_0.shape, nk_1.shape, nk_2.shape)
lambda_x = np.linspace(10, 11, 100)
e_0 = (np.interp(lambda_x, nk_0[:, 0], nk_0[:, 1]) + 1j * np.interp(lambda_x, nk_0[:, 0], nk_0[:, 2])) ** 2
e_1 = (np.interp(lambda_x, nk_1[:, 0], nk_1[:, 1]) + 1j * np.interp(lambda_x, nk_1[:, 0], nk_1[:, 2])) ** 2
e_2 = (np.interp(lambda_x, nk_2[:, 0], nk_2[:, 1]) + 1j * np.interp(lambda_x, nk_2[:, 0], nk_2[:, 2])) ** 2

def eps_at(lam):
  e0 = np.interp(lam, lambda_x, e_0)
  e1 = np.interp(lam, lambda_x, e_1)
  e2 = np.interp(lam, lambda_x, e_2)
  # print([e0, e1, e2])
  return [e0, e1, e2]


# 归一化的网格坐标点
x_lin = np.linspace(0, 1, Nx)
y_lin = np.linspace(0, 1, Ny)
ptx, pty = np.meshgrid(x_lin, y_lin, indexing = 'ij')
ptx, pty = ptx.flatten(), pty.flatten()

# 构建图案，此处为在矩形图案中间挖掉一个圆
# 由于作用在归一化的 [0, 1] × [0, 1] 网格 ptx 和 pty 上，这里 a, b, r 分别为矩形长宽、圆半径的占空比
# a = b = 1 时矩形占满，r = 1 时为内切圆
def pattern(a, b, r):
  # 矩形使用相对切比雪夫距离
  fill1 = np.maximum(np.abs(ptx - 0.5) / 0.5 / max(a, 1e-7), np.abs(pty - 0.5) / 0.5 / max(b, 1e-7))
  fill1 = 1 / (1 + np.exp(233 * (np.minimum(2, fill1) - 1)))
  # 圆形使用相对欧几里得距离
  fill2 = np.sqrt((ptx - 0.5) ** 2 + (pty - 0.5) ** 2) / 0.5 / max(r, 1e-7)
  fill2 = 1 / (1 + np.exp(-233 * (np.minimum(2, fill2) - 1)))
  return np.minimum(fill1, fill2)

# 传入结构参数，返回反射率
# 假定光源只会改波长和角度吧，偏振固定sp混合
# para = [pd, t0, t1, a, b, r]
def rcwa_calc(lam, theta, para):
  pd, t0, t1, a, b, r = para[0], para[1], para[2], para[3], para[4], para[5]
  epsilon = eps_at(lam)
  ep = e_vacuum + pattern(a, b, r) * (epsilon[0] - e_vacuum)
  if (LOG_ON):
    fill = pattern(a, b, r)
    for i in range(0, Nx):
      for j in range(0, Ny):
        print(str('%.1f' % fill[i * Nx + j]), end = ' ', file = sys.stderr)
      print(file = sys.stderr)

  L1, L2 = [1, 0], [0, 1]
  freq, phi = 1 / lam * (1 + 1j / 2 / 1e9), 0
  obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose = 0)
  obj.Add_LayerUniform(100, e_vacuum)
  obj.Add_LayerGrid(t0, Nx, Ny)
  obj.Add_LayerUniform(t1, epsilon[1])
  obj.Add_LayerUniform(1, epsilon[2]) # 设一个很厚的金属反射背板
  obj.Add_LayerUniform(100, e_vacuum)
  obj.Init_Setup(Pscale = pd)
  obj.MakeExcitationPlanewave(plane_wv['p_amp'], plane_wv['p_phase'], plane_wv['s_amp'], plane_wv['s_phase'], order = 0)
  obj.GridLayer_geteps(ep)
  R, T = obj.RT_Solve(normalize = 1)
  if (LOG_ON):
    print('R = ' + str(R) + ', T = ' + str(T), file = sys.stderr)
  return np.abs(R)


# 给定入射角度与结构参数，计算光谱
def get_spec(theta, para):
  spec = np.zeros(lambda_x.size)
  for i in range(0, lambda_x.size):
    spec[i] = rcwa_calc(lambda_x[i], theta, para)
  return spec


# 目标函数，参量为优化变量
def func_target(para):
  # print('type of x :', type(x), ', x =', x)
  # 假设现在的优化目标是反射率最接近0.5
  theta = 0
  max_R = 0
  for lam in lambda_x:
    max_R = max(max_R, np.abs(rcwa_calc(lam, theta, para) - 0.5))
  return max_R

counter = 0
grad_target = autograd.grad(func_target)

# 优化函数，参量包括优化变量及其梯度
def func_nlopt(x, grad_x):
  global counter
  grad_x[:] = grad_target(x)
  y = func_target(x)
  print('Step =', counter, ', para =', x, ', R =', y)
  print('Step =', counter, ', para =', x, ', R =', y, file = sys.stderr)
  counter += 1
  return y



if __name__ == '__main__':
  # 将标准错误流重定向到 log.txt
  saved_stderr = sys.stderr
  log_file = open("log.txt", "w")
  sys.stderr = log_file
  time_start = time.perf_counter()
  # ---------------------------


  if (True):
    my_draw.draw(lambda_x, get_spec(0, [10, 4, 9, 0.5, 0.3, 0.2]), 'reflect - rcwa')
    plt.ylim(0, 1)
    plt.show()
    # exit()

  print('start opt...')
  lower_bd = np.array([1, 1, 1, 0, 0, 0])
  upper_bd = np.array([20, 10, 10, 1, 1, 0.8])
  opt = nlopt.opt(nlopt.LD_MMA, 6)
  opt.set_lower_bounds(lower_bd)
  opt.set_upper_bounds(upper_bd)
  opt.set_xtol_rel(1e-5)
  opt.set_maxeval(100)
  opt.set_min_objective(func_nlopt)

  init_x = np.array([12, 3, 10, 0.9, 0.6, 0.2])
  best_x = init_x
  for rd in range(0, 3):
    print('round:', rd)
    print('round:', rd, file = sys.stderr)
    counter = 0
    b_x = opt.optimize(init_x)
    if (func_target(b_x) < func_target(best_x)):
      best_x = b_x
    init_x = np.random.random(6) * (upper_bd - lower_bd) + lower_bd

  print('beat =', best_x)
  print('best =', best_x, file = sys.stderr)

  time_end = time.perf_counter()
  print('Running time: %s Seconds' % (time_end - time_start))

  my_draw.draw(lambda_x, get_spec(0, best_x), 'reflect - rcwa')
  plt.ylim(0, 1)
  nG = int(nG * 1.6)
  my_draw.draw(lambda_x, get_spec(0, best_x), 'reflect - rcwa - greater nG')
  plt.ylim(0, 1)
  plt.show()