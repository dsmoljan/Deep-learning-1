import matplotlib.pyplot as plt 
import matplotlib
from matplotlib.pyplot import figure
import numpy as np


def eval_perf_binary(Y, Y_):
    tp = sum(np.logical_and(Y == Y_, Y_ == True))
    fn = sum(np.logical_and(Y != Y_, Y_ == True))
    tn = sum(np.logical_and(Y == Y_, Y_ == False))
    fp = sum(np.logical_and(Y != Y_, Y_ == False))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    f1 = tp / (tp + 0.5 * (tp + fn))
    conf_matrix = [tp, fn, tn, fp]
    return accuracy, recall, precision, f1, conf_matrix


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    linewidth = 2
    legend_size = 10
    train_color = "m"
    val_color = "c"

    num_points = len(data["train_loss"])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title("Cross-entropy loss")
    ax1.plot(
        x_data,
        data["train_loss"],
        marker="o",
        color=train_color,
        linewidth=linewidth,
        linestyle="-",
        label="train",
    )
    ax1.plot(
        x_data,
        data["valid_loss"],
        marker="o",
        color=val_color,
        linewidth=linewidth,
        linestyle="-",
        label="validation",
    )
    ax1.legend(loc="upper right", fontsize=legend_size)
    ax2.set_title("Average accuracy")
    ax2.plot(
        x_data,
        data["train_acc"],
        marker="o",
        color=train_color,
        linewidth=linewidth,
        linestyle="-",
        label="train",
    )
    ax2.plot(
        x_data,
        data["valid_acc"],
        marker="o",
        color=val_color,
        linewidth=linewidth,
        linestyle="-",
        label="validation",
    )
    ax2.legend(loc="upper left", fontsize=legend_size)
    ax3.set_title("Learning rate")
    ax3.plot(
        x_data,
        data["lr"],
        marker="o",
        color=train_color,
        linewidth=linewidth,
        linestyle="-",
        label="learning_rate",
    )
    ax3.legend(loc="upper left", fontsize=legend_size)

    save_path = os.path.join(save_dir, "training_plot.png")
    print("Plotting in: ", save_path)
    plt.savefig(save_path)

def draw_table(column_headers, row_headers, cell_text, title):
  rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
  ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

  matplotlib.rcParams["figure.dpi"] = 200

  fig, ax = plt.subplots() 
  ax.set_axis_off() 
  table = ax.table( 
      cellText = cell_text,  
      rowLabels = row_headers,  
      colLabels = column_headers, 
      rowColours = rcolors,  
      colColours = ccolors, 
      cellLoc ='center',  
      loc ='upper left')     
    
  ax.set_title(title,fontweight ="bold") 
    
  plt.show()
