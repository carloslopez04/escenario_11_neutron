import numpy as np
import math as m

class neurona:
  def __init__(self,peso,sesgo,func):
    self.peso = np.array(peso)
    self.sesgo = sesgo
    self.func = func

  def activacion(self,x):
    if self.func == "ReLu":
      return max(0, np.array(x))
    elif self.func == "Sigmoid":
      return 1 / (1 + m.exp(-x))
    elif self.func == "Tanh":
      return np.tanh(x)
    elif self.func == "Binary_step":
      return 1 if x >= 0 else 0
    else:
      raise ValueError(f"Función de activación '{self.func}' no es válida.")

  def run(self,entrada):
    self.entrada = np.array(entrada)
    calculo = np.dot(self.peso,entrada) + self.sesgo
    return self.activacion(calculo)

  def cambiosesgo(self,nsesgo):
    self.sesgo = nsesgo