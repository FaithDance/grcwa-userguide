import matplotlib.pyplot as plt

def draw(x, y, title):
  plt.figure(figsize = (8, 5))
  plt.plot(x, y)
  plt.title(title)

def draw_nk(x, n, k, title):
  plt.figure(figsize = (8, 5))
  plt.plot(x, n, label = 'n')
  plt.plot(x, k, label = 'k')
  plt.title(title)
  plt.legend()