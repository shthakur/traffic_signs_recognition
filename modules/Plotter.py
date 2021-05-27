import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def plot_curve(self, sub_plot, params, train_column, valid_column,
                   linewidth=2, train_linestyle="b-", valid_linestyle="g-"):
        train_values = params[train_column]
        valid_values = params[valid_column]
        epochs = train_values.shape[0]
        x_axis = np.arange(epochs)
        sub_plot.plot(x_axis[train_values > 0], train_values[train_values > 0],
                      train_linestyle, linewidth=linewidth, label="train")
        sub_plot.plot(x_axis[valid_values > 0], valid_values[valid_values > 0],
                      valid_linestyle, linewidth=linewidth, label="valid")
        return epochs

    # Plot history curves
    def plot_learning_curves(self, params):
        curves_figure = plt.figure(figsize=(10, 4))
        sub_plot = curves_figure.add_subplot(1, 2, 1)
        epochs_plotted = self.plot_curve(
            sub_plot, params, train_column="train_acc", valid_column="val_acc")

        plt.grid()
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.xlim(0, epochs_plotted)

        sub_plot = curves_figure.add_subplot(1, 2, 2)
        epochs_plotted = self.plot_curve(
            sub_plot, params, train_column="train_loss", valid_column="val_loss")

        plt.grid()
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.xlim(0, epochs_plotted)
        plt.yscale("log")


plotter = Plotter()
