import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
# Use 'Agg' so this program could run on a remote server
# matplotlib.use('Agg')
# style.use('ggplot')
sns.set_style(style="darkgrid")

p_path = 'p.npy'
r_path = 'r.npy'
x = np.load(r_path)
y = np.load(p_path)
auc = sklearn.metrics.auc(x=x, y=y)
plt.plot(x, y, lw=1.0, markevery=200, marker='d', label='CIL | AUC: 0.508')

plt.xlabel('recall')
plt.ylabel('precision')
plt.ylim([0.3, 1.0])
plt.xlim([0.0, 0.7])
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('pr.png', bbox_inches='tight')
plt.savefig('pr.pdf', bbox_inches='tight')
