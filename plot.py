import matplotlib.pyplot as plt
import numpy as np
from util import load_result

"""
Visualizetion of experiment result
"""


def plot_func(exp_list, color, line):
	fig = plt.figure(figsize=(8, 8))

	# plot ppl
	plt.subplot(2, 1, 1)
	plt.xlim(0, 4800)
	plt.xlabel('Number of updates')
	plt.ylim(0, 1000)
	plt.ylabel('valid ppl')
	for i in range(len(exp_list)):
		experiment = np.array(exp_list[i])
		plt.plot(experiment[:, 0], experiment[:, 1], color[i], label=line[i])

	plt.legend()

	# plot acc
	plt.subplot(2, 1, 2)
	plt.xlim(0, 4800)
	plt.xlabel('Number of updates')
	plt.ylim(0, 100)
	plt.ylabel('valid acc (%)')
	for i in range(len(exp_list)):
		experiment = np.array(exp_list[i])
		plt.plot(experiment[:, 0], experiment[:, 2]*100, color[i], label=line[i])

	plt.legend()
	plt.show()

def plot_n_16():
	# input size = 16, standard
	exp_list = []
	exp_list.append(load_result(input_size=16, full_attention=True))
	# input size = 16, k=8
	exp_list.append(load_result(input_size=16, full_attention=False, dim_k=8))
	# input size = 16, k=4
	exp_list.append(load_result(input_size=16, full_attention=False, dim_k=4))

	color = ['r--', 'b-', 'g-.']
	line = ['standard transformer, n=16', 'Linformer, n=16, k=8', 'Linformer, n=16, k=4']

	plot_func(exp_list, color, line)

def plot_parameter_sharing():
	exp_list = []
	exp_list.append(load_result(custom_path="N_16_k_8_layerwise"))
	# input size = 16, k=8
	exp_list.append(load_result(custom_path="N_16_k_8_headwise"))
	# input size = 16, k=4
	exp_list.append(load_result(custom_path="N_16_k_8_kv"))

	color = ['r--', 'b-', 'g-.']
	line = ['Layerwise sharing', 'Headwise sharing', 'Key-value sharing']

	plot_func(exp_list, color, line)


def main():
	plot_n_16()
	plot_parameter_sharing()

if __name__ == '__main__':
    main()
